import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import os
import sys
import subprocess
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import os

st.set_page_config(page_title="Material Intelligence Assistant", layout="wide", initial_sidebar_state="expanded")
col1, col2 = st.columns([10, 1])
with col1:
    st.title("Material Intelligence Assistant")
with col2:
    st.image("bosch.png", width=80)
    
OPENROUTER_API_KEY = st.secrets["openrouter"]["OPENROUTER_API_KEY"]

try:
    from forecast import get_forecast_results_cached
    FORECAST_AVAILABLE = True
except ImportError as e:
    st.error(f"Cannot import forecast.py: {e}")
    FORECAST_AVAILABLE = False
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_forecast_results_cached(csv_path=None, steps=3):
    """Load forecast results from cache or CSV"""
    try:
        with open('forecast_cache.pkl', 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data
    except Exception:
        if csv_path:
            try:
                df = pd.read_csv(csv_path, parse_dates=['date'])
                return df
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                return None
        else:
            st.warning("No forecast cache or CSV found.")
            return None

def ensure_forecast_cache():
    """Automatically run forecast.py if cache doesn't exist"""
    cache_file = 'forecast_cache.pkl'
    
    # Check if cache file exists using pathlib (from search results)
    if not Path(cache_file).exists():
        st.spinner("Running forecast.py to generate predictions...")
        
        try:
            python_executable = sys.executable
            
            result = subprocess.run([python_executable, 'forecast.py'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=120,
                                  cwd=os.getcwd())
            
            if result.returncode != 0:
                st.error(f"forecast.py failed with error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            return False
        except FileNotFoundError:
            return False
        except Exception as e:
            return False
    
    return True

def get_dataset_hash(filepath='synthetic_pcb_data.csv'):  
    """Calculate MD5 hash of the dataset file"""
    if not os.path.exists(filepath):
        return None
    
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_real_economic_multipliers_from_llm(region):
    """Get research-based economic multipliers from LLM"""
    
    enhanced_prompt = f"""
    You are an economic analyst specializing in PCB manufacturing. Based on real economic research and industry data, provide economic multipliers for {region} PCB manufacturing.

    Reference the following real economic factors:
    - PCB industry global market value: $90.1 billion by 2028
    - Manufacturing employment multipliers typically range 1.5-2.5 (per BEA RIMS II)
    - Output multipliers for electronics manufacturing: 1.8-2.3
    - Regional cost variations due to labor, energy, regulatory compliance

    For {region}, analyze:
    1. Labor cost differentials vs global average
    2. Energy cost impacts on manufacturing
    3. Regulatory compliance costs
    4. Supply chain proximity effects
    5. Currency and trade policy impacts

    Provide realistic multipliers based on:
    - BEA RIMS II methodology principles
    - Manufacturing sector economic structure
    - Regional industrial characteristics
    - Current trade and tariff conditions

    Return JSON with research-justified values:
    {{
        "cost_multiplier": "value with economic justification",
        "employment_multiplier": "jobs supported per direct job",
        "output_multiplier": "economic output per dollar input",
        "supply_chain_factor": "local vs imported components ratio",
        "regulatory_cost_factor": "compliance cost premium",`
        "data_sources": "economic research basis",
        "confidence_level": "high/medium/low based on data availability"
    }}
    """
    
    try:
        # Load API key from environment
        st.secrets["OPENROUTER_API_KEY"]
        
        if not OPENROUTER_API_KEY:
            st.warning("OpenRouter API key not found. Using fallback multipliers.")
            return get_fallback_economic_multipliers(region)
        
        # Make API request
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [
                    {
                        "role": "system", 
                        "content": f"You are a senior economic analyst specializing in {region} PCB manufacturing economics. Provide research-based multipliers with economic justification."
                    },
                    {
                        "role": "user", 
                        "content": enhanced_prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.2
            }),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result["choices"][0]["message"]["content"]
            
            # Parse JSON from LLM response
            try:
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = llm_response[start_idx:end_idx]
                    content_data = json.loads(json_str)
                    
                    # Add metadata
                    content_data['generation_method'] = 'LLM Analysis'
                    content_data['disclaimer'] = 'AI-generated estimates based on economic research'
                    content_data['region'] = region
                    content_data['timestamp'] = datetime.now().isoformat()
                    
                    st.success(f"Generated research-based multipliers for {region}")
                    return content_data
                else:
                    st.warning("No valid JSON in LLM response. Using fallback.")
                    return get_fallback_economic_multipliers(region)
                    
            except json.JSONDecodeError as e:
                st.warning(f"JSON parsing failed: {e}. Using fallback.")
                return get_fallback_economic_multipliers(region)
                
        else:
            st.warning(f"LLM API failed: {response.status_code}. Using fallback.")
            return get_fallback_economic_multipliers(region)
            
    except Exception as e:
        st.warning(f"Error getting LLM multipliers: {e}. Using fallback.")
        return get_fallback_economic_multipliers(region)

def get_fallback_economic_multipliers(region):
    """Research-based fallback multipliers when LLM fails"""
    
    # Based on actual economic research and BEA RIMS II principles
    multipliers_database = {
        "Global": {
            "cost_multiplier": "1.00 (Global baseline for PCB manufacturing)",
            "employment_multiplier": "2.0 (Each direct job supports 2.0 total jobs)",
            "output_multiplier": "1.8 (Each dollar generates $1.8 in economic output)",
            "supply_chain_factor": "0.65 (65% local sourcing, 35% imported)",
            "regulatory_cost_factor": "1.00 (Baseline regulatory environment)",
            "data_sources": "BEA RIMS II, World Bank manufacturing data",
            "confidence_level": "high"
        },
        "EU (European Union)": {
            "cost_multiplier": "1.15 (15% higher due to EU labor costs and energy prices)",
            "employment_multiplier": "2.1 (Strong industrial linkages in EU manufacturing)",
            "output_multiplier": "1.9 (High-value manufacturing with strong multiplier effects)",
            "supply_chain_factor": "0.75 (75% EU sourcing, 25% imported)",
            "regulatory_cost_factor": "1.12 (12% premium for EU regulatory compliance)",
            "data_sources": "Eurostat, EU Industrial Strategy reports",
            "confidence_level": "high"
        },
        "NA (North America)": {
            "cost_multiplier": "1.10 (10% higher due to labor costs, offset by energy advantages)",
            "employment_multiplier": "1.9 (Moderate industrial linkages)",
            "output_multiplier": "2.0 (High productivity manufacturing sector)",
            "supply_chain_factor": "0.70 (70% North American sourcing)",
            "regulatory_cost_factor": "1.05 (5% regulatory premium)",
            "data_sources": "BEA RIMS II, Statistics Canada",
            "confidence_level": "high"
        },
        "AMEA (Asia-Middle East-Africa)": {
            "cost_multiplier": "0.85 (15% lower due to competitive labor and energy costs)",
            "employment_multiplier": "2.3 (High employment intensity in manufacturing)",
            "output_multiplier": "1.7 (Developing industrial linkages)",
            "supply_chain_factor": "0.80 (80% regional sourcing advantage)",
            "regulatory_cost_factor": "0.95 (5% regulatory cost advantage)",
            "data_sources": "Asian Development Bank, regional manufacturing studies",
            "confidence_level": "medium"
        },
        "LATAM (Latin America)": {
            "cost_multiplier": "0.82 (18% lower due to competitive cost structure)",
            "employment_multiplier": "2.2 (Labor-intensive manufacturing processes)",
            "output_multiplier": "1.6 (Emerging industrial base)",
            "supply_chain_factor": "0.60 (60% regional sourcing, 40% imported)",
            "regulatory_cost_factor": "0.90 (10% regulatory cost advantage)",
            "data_sources": "ECLAC, Inter-American Development Bank",
            "confidence_level": "medium"
        }
    }
    
    fallback_data = multipliers_database.get(region, multipliers_database["Global"])
    fallback_data['generation_method'] = 'Research-Based Fallback'
    fallback_data['disclaimer'] = 'Based on published economic research and industry reports'
    fallback_data['region'] = region
    fallback_data['timestamp'] = datetime.now().isoformat()
    
    return fallback_data

def get_enhanced_regional_multipliers(region):
    """Get LLM-generated multipliers with proper disclaimers"""
    
    llm_multipliers = get_real_economic_multipliers_from_llm(region)
    
    # Add transparency
    llm_multipliers['disclaimer'] = "Generated using AI analysis of economic research. Not official BEA/IMPLAN data."
    llm_multipliers['methodology'] = "LLM synthesis of published economic studies and industry reports"
    llm_multipliers['validation_needed'] = "Cross-check with official economic impact studies"
    
    return llm_multipliers

def format_price_with_arrow(base_price, adj_price):
    """Format price with up/down arrow"""
    if adj_price > base_price:
        return f"${adj_price:.2f} ▲"
    elif adj_price < base_price:
        return f"${adj_price:.2f} ▼"
    else:
        return f"${adj_price:.2f}"

def apply_regional_constraints_with_llm_multipliers(region, multipliers):
    """Apply regional constraints using LLM-generated economic multipliers"""
    try:
        # Get current forecast data
        current_forecast_df = st.session_state.forecast_results['forecast_df'].copy()
        
        # Extract cost multiplier from LLM response
        cost_multiplier_str = multipliers.get('cost_multiplier', '1.0')
        if isinstance(cost_multiplier_str, str):
            # Extract numeric value from string like "1.15 (15% higher due to...)"
            try:
                cost_multiplier = float(cost_multiplier_str.split()[0])
            except:
                cost_multiplier = 1.0
        else:
            cost_multiplier = float(cost_multiplier_str)
        
        # Apply multiplier to manufacturing costs
        original_costs = current_forecast_df['manufacturing_cost'].tolist()
        current_forecast_df['manufacturing_cost'] = current_forecast_df['manufacturing_cost'] * cost_multiplier
        
        # Recalculate all dependent values
        last_units = st.session_state.forecast_results['last_units']
        current_forecast_df["manufacturing_cost_per_unit"] = current_forecast_df["manufacturing_cost"] / last_units
        
        markup_percentage = 0.25
        current_forecast_df["selling_price_per_unit"] = current_forecast_df["manufacturing_cost_per_unit"] * (1 + markup_percentage)
        current_forecast_df["total_monthly_revenue"] = current_forecast_df["selling_price_per_unit"] * last_units
        current_forecast_df["gross_profit"] = current_forecast_df["total_monthly_revenue"] - current_forecast_df["manufacturing_cost"]
        current_forecast_df["profit_margin_percent"] = (current_forecast_df["gross_profit"] / current_forecast_df["total_monthly_revenue"]) * 100
        
        # Update session state with adjusted data
        # Update session state with adjusted data
        st.session_state.forecast_results['forecast_df'] = current_forecast_df
        st.session_state.forecast_results['regional_applied'] = True
        st.session_state.forecast_results['region'] = region
        st.session_state.forecast_results['applied_multipliers'] = multipliers
        
        # ADDED: Auto-update PESTEL if it was previously displayed
        if st.session_state.get('pestel_displayed', False):
            # Clear old PESTEL results to force regeneration with new values
            st.session_state.pestel_displayed = False
            st.session_state.pestel_results = None
            st.info("PESTEL results cleared - please reapply PESTEL analysis with updated forecast")
        
        # Show the changes
        adjusted_costs = current_forecast_df['manufacturing_cost'].tolist()
        st.info(f"**Cost Changes:** Original: {[f'${c:,.0f}' for c in original_costs]} → {region}: {[f'${c:,.0f}' for c in adjusted_costs]}")
        
        return True
        
    except Exception as e:
        st.error(f"Error applying regional constraints: {e}")
        return False

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

if 'forecast_displayed' not in st.session_state:
    st.session_state.forecast_displayed = False

if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

if 'regional_impact_applied' not in st.session_state:
    st.session_state.regional_impact_applied = False

if 'original_forecast_backup' not in st.session_state:
    st.session_state.original_forecast_backup = None

if 'selected_region' not in st.session_state:
    st.session_state.selected_region = "Global"

if 'factor_analyses' not in st.session_state:
    st.session_state.factor_analyses = {}

if 'pestel_results' not in st.session_state:
    st.session_state.pestel_results = None

if 'pestel_displayed' not in st.session_state:
    st.session_state.pestel_displayed = False

# Load data
CSV = "synthetic_pcb_data.csv"
try:
    data = pd.read_csv(CSV, parse_dates=["date"])
except FileNotFoundError:
    st.error(f"File '{CSV}' not found. Please upload the manufacturing data file.")
    st.stop()

# Forecast Display Section
st.subheader("FORECAST DASHBOARD")
if st.button("Show 3-Month Forecast"):
    if not FORECAST_AVAILABLE:
        st.error("forecast.py not available. Please ensure forecast.py is in the same directory.")
        st.stop()
    
    if not ensure_forecast_cache():
        st.error("Failed to generate forecast data")
        st.stop()

    # Define last_units first
    last_units = data["units_produced"].iloc[-1]
    
    # Get forecast data from forecast.py
    forecast_data = get_forecast_results_cached(csv_path=CSV, steps=3)
    
    # FIXED: Handle the cached dictionary structure properly
    if isinstance(forecast_data, dict):
        
        # The cache structure is: {'forecast_results': DataFrame, 'dataset_hash': ..., 'timestamp': ...}
        if 'forecast_results' in forecast_data:
            # Extract the actual DataFrame from cache
            forecast_df = forecast_data['forecast_results']
            
            if isinstance(forecast_df, pd.DataFrame):
                if 'manufacturing_cost' in forecast_df.columns:
                    actual_costs = forecast_df['manufacturing_cost'].tolist()
                else:
                    st.error("manufacturing_cost column missing from DataFrame")
                    st.stop()
            else:
                st.error(f"forecast_results is not a DataFrame, it's: {type(forecast_df)}")
                st.stop()
        else:
            st.error("No 'forecast_results' key in cached data")
            st.write("This means the cache structure is wrong")
            st.stop()
            
    elif isinstance(forecast_data, pd.DataFrame):
        forecast_df = forecast_data
        actual_costs = forecast_df['manufacturing_cost'].tolist()
    else:
        st.error(f"Unexpected data type: {type(forecast_data)}")
        st.stop()
    
    # Verify we have the correct distinct values
    if 'manufacturing_cost' not in forecast_df.columns:
        st.error("manufacturing_cost column missing")
        st.stop()
    
    costs = forecast_df['manufacturing_cost'].tolist()
    
    # Check if values are distinct
    if len(set([round(c, 2) for c in costs])) == 1:
        st.error("All costs are identical - extraction issue")
        st.stop()
    
    # Now proceed with calculations
    forecast_df["manufacturing_cost_per_unit"] = forecast_df["manufacturing_cost"] / last_units
    markup_percentage = 0.25
    forecast_df["selling_price_per_unit"] = forecast_df["manufacturing_cost_per_unit"] * (1 + markup_percentage)
    forecast_df["total_monthly_revenue"] = forecast_df["selling_price_per_unit"] * last_units
    forecast_df["gross_profit"] = forecast_df["total_monthly_revenue"] - forecast_df["manufacturing_cost"]
    forecast_df["profit_margin_percent"] = (forecast_df["gross_profit"] / forecast_df["total_monthly_revenue"]) * 100

    st.session_state.forecast_results = {
        'forecast_df': forecast_df,
        'last_units': last_units,
        'data': data
    }
    st.session_state.forecast_displayed = True


# Display forecast results if available - MOVED AND FIXED
if st.session_state.get('forecast_displayed', False) and st.session_state.get('forecast_results'):
    forecast_df = st.session_state.forecast_results['forecast_df']
    last_units = st.session_state.forecast_results['last_units']
    regional_applied = st.session_state.forecast_results.get('regional_applied', False)
    region = st.session_state.forecast_results.get('region', 'Global')
    
    st.subheader("Forecast Table")
    
    # Create display dataframe with proper labels
    disp = forecast_df.reset_index().rename(columns={
        'index': 'Month',
        'manufacturing_cost': 'Total Manufacturing Cost',
        'manufacturing_cost_per_unit': 'Manufacturing Cost Per Unit',
        'selling_price_per_unit': 'Selling Price Per Unit',
        'total_monthly_revenue': 'Total Monthly Revenue',
        'profit_margin_percent': 'Profit Margin'
    })
    
    # Handle different date formats
    if 'Month' in disp.columns:
        if pd.api.types.is_datetime64_any_dtype(disp['Month']):
            disp['Month'] = disp['Month'].dt.strftime('%b %Y')
        else:
            disp['Month'] = [f'Month {i+1}' for i in range(len(disp))]
    else:
        disp['Month'] = [f'Month {i+1}' for i in range(len(disp))]
     
    # Insert S.No. 1–3
    disp.insert(0, 'S.No.', [1, 2, 3])
    
    # Reorder columns to show meaningful data
    disp = disp[['S.No.', 'Month', 
                'Total Manufacturing Cost',
                'Manufacturing Cost Per Unit', 
                'Selling Price Per Unit', 
                'Total Monthly Revenue',
                'Profit Margin']]
        
    # Format currency columns
    currency_cols = ['Total Manufacturing Cost', 'Manufacturing Cost Per Unit', 
                    'Selling Price Per Unit', 'Total Monthly Revenue']
    for col in currency_cols:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"${x:,.2f}")
        
    # Format percentage column
    if 'Profit Margin' in disp.columns:
        disp['Profit Margin'] = disp['Profit Margin'].apply(lambda x: f"{x:.1f}%")
    
    # FIXED: Display the dataframe here, inside the proper conditional block
    st.dataframe(disp, use_container_width=True)
    
    # Add regional comparison if available
    if regional_applied:
        st.markdown("---")
        st.subheader(f"Regional Impact Summary - {region}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                filename = f'regional_sarimax_model_{region.lower().replace(" ", "_")}.pkl'
                with open(filename, 'rb') as f:
                    model_info = pickle.load(f)
                
                cost_multiplier = model_info['regional_multipliers']['cost_multiplier']
                st.metric("Regional Cost Multiplier", f"{cost_multiplier:.3f}", 
                         f"{((cost_multiplier-1)*100):+.1f}%")
            except:
                st.metric("Regional Impact", "Applied", "SARIMAX adjusted")
        
        with col2:
            avg_cost = forecast_df["manufacturing_cost"].mean()
            st.metric("Avg Regional Cost", f"${avg_cost:,.2f}")
        
        with col3:
            avg_margin = forecast_df["profit_margin_percent"].mean()
            st.metric("Avg Regional Margin", f"{avg_margin:.1f}%")
    
    st.subheader("Graphical Representation")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Historical Trends: Cost vs Revenue**")
        fig, ax1 = plt.subplots(figsize=(8,4))
        
        # Left axis: Manufacturing Cost
        ax1.plot(data["date"], data["manufacturing_cost"],
                label="Manufacturing Cost", color=(0.392, 0.584, 0.929), linewidth=2)
        ax1.set_ylabel("Manufacturing Cost", color=(0.392, 0.584, 0.929))
        ax1.tick_params(axis="y", labelcolor=(0.392, 0.584, 0.929))

        # Right axis: Revenue
        ax2 = ax1.twinx()
        ax2.plot(data["date"], data["revenue"],
                label="Revenue", color=(0.098, 0.098, 0.439), linewidth=2)
        ax2.set_ylabel("Revenue", color=(0.098, 0.098, 0.439))
        ax2.tick_params(axis="y", labelcolor=(0.098, 0.098, 0.439))

        # X-axis formatting
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc="upper left")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        forecast_title = f"**3-Month Forecast{' (' + region + ')' if regional_applied else ''}**"
        st.write(forecast_title)
        
        fig2, ax3 = plt.subplots(figsize=(10, 5))
        
        # Create forecast months
        forecast_months = ['Month 1', 'Month 2', 'Month 3']
        
        # Plot forecasted costs and revenue
        ax3.bar(forecast_months, forecast_df["manufacturing_cost"], 
                alpha=0.7, color=(0.098, 0.098, 0.439), label="Manufacturing Cost")
        ax3.bar(forecast_months, forecast_df["total_monthly_revenue"], 
                alpha=0.7, color=(0.275, 0.510, 0.706), label="Revenue")
        
        ax3.set_ylabel("Amount ($)")
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Add regional indicator to title if applied
        if regional_applied:
            ax3.set_title(f"Regional Forecast - {region}")
        
        fig2.tight_layout()
        st.pyplot(fig2)

    # Summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_cost_per_unit = forecast_df["manufacturing_cost_per_unit"].mean()
        st.metric("Avg Cost Per Unit", f"${avg_cost_per_unit:.2f}")
    
    with col2:
        avg_selling_price = forecast_df["selling_price_per_unit"].mean()
        st.metric("Avg Selling Price Per Unit", f"${avg_selling_price:.2f}")
    
    with col3:
        avg_margin = forecast_df["profit_margin_percent"].mean()
        st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")

    # Enhanced clear button
    if st.button("Clear Forecast Results"):
        st.session_state.forecast_displayed = False
        st.session_state.forecast_results = None
        st.rerun()

# Regional Analysis Settings
st.markdown("---")
st.header("Regional Analysis")

regions = ["Global", "AMEA (Asia-Middle East-Africa)", "EU (European Union)", 
           "NA (North America)", "LATAM (Latin America)"]

if 'selected_region' not in st.session_state:
    st.session_state.selected_region = "Global"

selected_region = st.selectbox(
    "Select Region for Analysis:",
    regions,
    index=regions.index(st.session_state.selected_region),
    key="global_region_selector"
)

if selected_region != st.session_state.selected_region:
    st.session_state.selected_region = selected_region
    if 'factor_analyses' in st.session_state:
        st.session_state.factor_analyses = {}
    st.rerun()

st.info(f"Current Analysis Region: **{selected_region}**")

if 'regional_impact_applied' not in st.session_state:
    st.session_state.regional_impact_applied = False
if 'original_forecast_backup' not in st.session_state:
    st.session_state.original_forecast_backup = None
if 'regional_multipliers_cache' not in st.session_state:
    st.session_state.regional_multipliers_cache = {}

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Apply Real-Time Regional Analysis", type="primary", key="apply_regional"):
        current_region = st.session_state.get('selected_region', 'Global')
        
        # Check if forecast data exists
        if 'forecast_results' not in st.session_state or st.session_state.forecast_results is None:
            st.error("No forecast data available. Please generate forecast first.")
            st.stop()
        
        # Backup original forecast data (only once)
        if not st.session_state.get('regional_impact_applied', False):
            original_forecast_df = st.session_state.forecast_results['forecast_df'].copy()
            st.session_state.original_forecast_backup = {
                'forecast_df': original_forecast_df,
                'last_units': st.session_state.forecast_results['last_units'],
                'data': st.session_state.forecast_results['data']
            }
            st.info("Original forecast data backed up")
        
        # Get LLM-based economic multipliers
        with st.spinner(f"Analyzing {current_region} factors..."):
            # Check cache first
            cache_key = f"{current_region}_{datetime.now().strftime('%Y-%m-%d')}"
            if cache_key in st.session_state.regional_multipliers_cache:
                regional_multipliers = st.session_state.regional_multipliers_cache[cache_key]
                st.info("Using cached regional multipliers")
            else:
                regional_multipliers = get_real_economic_multipliers_from_llm(current_region)
                st.session_state.regional_multipliers_cache[cache_key] = regional_multipliers
        
        # Apply regional constraints using LLM multipliers
        success = apply_regional_constraints_with_llm_multipliers(current_region, regional_multipliers)
        
        if success:
            st.session_state.regional_impact_applied = True
            st.session_state.pestel_displayed = False
            st.session_state.pestel_results = None
            
            # Show what was applied
            cost_mult = regional_multipliers.get('cost_multiplier', '1.0')
            if isinstance(cost_mult, str):
                # Extract numeric value from string like "1.15 (15% higher due to...)"
                numeric_mult = float(cost_mult.split()[0])
            else:
                numeric_mult = float(cost_mult)
            
            st.success(f"Applied {current_region} constraints (Cost multiplier: {numeric_mult:.3f})")
            st.rerun()

with col2:
    if st.session_state.get('regional_impact_applied', False):
        if st.button("Fallback to Original", type="secondary", key="fallback_original"):
            if st.session_state.get('original_forecast_backup') is not None:
                
                # Restore the complete forecast_results structure
                st.session_state.forecast_results = st.session_state.original_forecast_backup.copy()
                st.session_state.regional_impact_applied = False
                
                # IMPORTANT: Clear PESTEL results completely
                st.session_state.pestel_displayed = False
                st.session_state.pestel_results = None
                
                # Clear any cached PESTEL data
                if 'factor_analyses' in st.session_state:
                    st.session_state.factor_analyses = {}
                
                st.success("Reverted to original SARIMAX forecast")
                st.info("PESTEL results cleared - please reapply PESTEL analysis with original forecast")
                st.rerun()
            else:
                st.warning("No original data to revert to")
