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
import time
import re
from datetime import datetime
from fpdf import FPDF
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    st.error("API key not loaded from .env file")

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
        import pickle
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
        "regulatory_cost_factor": "compliance cost premium",
        "data_sources": "economic research basis",
        "confidence_level": "high/medium/low based on data availability"
    }}
    """
    
    try:
        # Load API key from environment
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        
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
                "model": "mistralai/mistral-7b-instruct:free",
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

def calculate_dynamic_monthly_prices_updated(base_costs, factor_analyses, region):
    def get_dynamic_pestel_impact_from_llm(region, factor_analyses):
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        
        if not OPENROUTER_API_KEY:
            regional_economic_data = {
                "Global": {"inflation_rate": 0.035, "regulatory_burden": 0.02, "political_stability": 0.01, "tech_adoption": -0.005},
                "EU (European Union)": {"inflation_rate": 0.025, "regulatory_burden": 0.045, "political_stability": 0.005, "tech_adoption": -0.01, "energy_costs": 0.03, "labor_costs": 0.02},
                "NA (North America)": {"inflation_rate": 0.03, "regulatory_burden": 0.025, "political_stability": 0.015, "tech_adoption": -0.015, "energy_costs": -0.005, "labor_costs": 0.025},
                "AMEA (Asia-Middle East-Africa)": {"inflation_rate": 0.045, "regulatory_burden": 0.015, "political_stability": 0.025, "tech_adoption": -0.02, "energy_costs": 0.02, "currency_volatility": 0.015},
                "LATAM (Latin America)": {"inflation_rate": 0.08, "regulatory_burden": 0.02, "political_stability": 0.035, "tech_adoption": -0.005, "energy_costs": 0.025, "currency_volatility": 0.025}
            }
            
            region_data = regional_economic_data.get(region, regional_economic_data["Global"])
            base_multiplier = 1.0
            impact_breakdown = {}
            
            for factor, impact in region_data.items():
                base_multiplier += impact
                impact_breakdown[factor] = impact
            
            manufacturing_multipliers = {"Global": 1.75, "EU (European Union)": 2.1, "NA (North America)": 1.9, "AMEA (Asia-Middle East-Africa)": 1.6, "LATAM (Latin America)": 1.4}
            manufacturing_multiplier = manufacturing_multipliers.get(region, 1.75)
            final_multiplier = 1.0 + ((base_multiplier - 1.0) * manufacturing_multiplier)
            final_multiplier = max(0.85, min(1.25, final_multiplier))
            
            st.info(f"**Dynamic Regional Impact for {region}:** {(final_multiplier-1)*100:+.1f}%")
            return final_multiplier
        
        context = f"Based on current analysis: {str(factor_analyses)[:500]}" if factor_analyses else ""
        
        prompt = f"""Calculate the total PESTEL impact multiplier for PCB manufacturing costs in {region} for 2025. {context}
        
        Consider: Political (trade policies, tariffs), Economic (inflation, currency), Social (labor), Technological (automation), Environmental (compliance), Legal (regulations).
        
        Respond ONLY with:
        MULTIPLIER: [number]"""
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                data=json.dumps({
                    "model": "mistralai/mistral-7b-instruct:free",
                    "messages": [{"role": "system", "content": f"You are a manufacturing cost analyst for {region}."}, {"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.2
                }),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                multiplier_match = re.search(r'MULTIPLIER:\s*([0-9.]+)', content)
                if multiplier_match:
                    multiplier = float(multiplier_match.group(1))
                    if 0.8 <= multiplier <= 1.3:
                        return multiplier
                
                percent_matches = re.findall(r'(\d+(?:\.\d+)?)%', content)
                if percent_matches:
                    percent = float(percent_matches[0])
                    if 0 <= percent <= 50:
                        multiplier = 1 + (percent / 100)
                        return multiplier
                
                return 1.08
            else:
                return 1.08
        except Exception as e:
            return 1.08
    
    pestel_impact = get_dynamic_pestel_impact_from_llm(region, factor_analyses)
    monthly_data = []
    
    for i, base_cost in enumerate(base_costs):
        adjusted_cost = base_cost * pestel_impact
        last_units = st.session_state.forecast_results.get('last_units', 2675)
        base_price = base_cost / last_units
        adjusted_price = adjusted_cost / last_units
        
        if adjusted_cost > base_cost:
            adjustment_direction = "Up"
        elif adjusted_cost < base_cost:
            adjustment_direction = "Down"
        else:
            adjustment_direction = "Same"
        
        monthly_data.append({
            'month': i + 1,
            'date': f'{["Jan", "Feb", "Mar"][i]} 2025',
            'base_cost': base_cost,
            'adjusted_cost': adjusted_cost,
            'base_price': base_price,
            'adjusted_price': adjusted_price,
            'adjustment_direction': adjustment_direction,
            'cost_change': adjusted_cost - base_cost,
            'price_change': adjusted_price - base_price
        })
    
    return monthly_data

def format_price_with_arrow(base_price, adj_price):
    """Format price with up/down arrow"""
    if adj_price > base_price:
        return f"${adj_price:.2f} ▲"
    elif adj_price < base_price:
        return f"${adj_price:.2f} ▼"
    else:
        return f"${adj_price:.2f}"

def generate_enhanced_regional_pdf_report(forecast_data, pestel_results):
    """Generate comprehensive PDF report with caching and enhanced formatting"""
    
    # Get the currently selected region from session state
    current_region = st.session_state.get('selected_region', 'Global')
    region_mapping = {
        "Global": "Global",
        "AMEA (Asia-Middle East-Africa)": "AMEA",
        "EU (European Union)": "EU", 
        "NA (North America)": "NA",
        "LATAM (Latin America)": "LATAM"
    }
    region_code = region_mapping.get(current_region, "Global")

    def get_pdf_cache_key(region, forecast_data):
        """Generate cache key based on region and forecast data"""
        forecast_hash = hashlib.md5(str(forecast_data).encode()).hexdigest()[:8]
        return f"pdf_content_cache_{region}_{forecast_hash}.pkl"
    
    def load_pdf_content_cache(cache_key):
        """Load cached PDF content if available"""
        try:
            if os.path.exists(cache_key):
                with open(cache_key, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is less than 1 hour old
                cache_time = cached_data.get('timestamp', 0)
                current_time = time.time()
                
                if current_time - cache_time < 3600:  # 1 hour cache
                    return cached_data.get('content')
                else:
                    os.remove(cache_key)  # Remove old cache
                    
        except Exception as e:
            st.warning(f"Cache loading failed: {e}")
        
        return None
    
    def save_pdf_content_cache(cache_key, content):
        """Save PDF content to cache"""
        try:
            cache_data = {
                'content': content,
                'timestamp': time.time(),
                'region': region_code
            }
            
            with open(cache_key, 'wb') as f:
                pickle.dump(cache_data, f)
            
        except Exception as e:
            st.warning(f"Cache saving failed: {e}")

    def clean_text_for_pdf(text):
        """Clean and format text for better PDF presentation"""
        if not text:
            return ""
        
        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            '€': 'EUR', '£': 'GBP', '¥': 'JPY', '₹': 'INR', '₽': 'RUB', '¢': 'cents',
            '°': ' degrees', '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"',
            '…': '...', '•': '-', '™': 'TM', '®': '(R)', '©': '(C)', '±': '+/-',
            '×': 'x', '÷': '/', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'µ': 'micro'
        }
        
        # Apply replacements
        for unicode_char, ascii_replacement in replacements.items():
            text = text.replace(unicode_char, ascii_replacement)
        
        # ENHANCED FORMATTING:
        # Remove markdown stars but preserve emphasis through formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)        # Remove *italic*
        
        # Improve numbered list formatting
        text = re.sub(r'\n\s*(\d+)\.\s*', r'\n\n\1. ', text)  # Add space before numbered items
        
        # Improve bullet point formatting
        text = re.sub(r'\n\s*[-•]\s*', r'\n\n- ', text)  # Add space before bullet points
        
        # Add proper spacing after colons in lists
        text = re.sub(r'(\d+\.\s*[^:]+):\s*', r'\1: ', text)
        
        # Ensure proper paragraph spacing
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 line breaks
        text = re.sub(r'([.!?])\s*\n([A-Z])', r'\1\n\n\2', text)  # Add space between sentences
        
        # Clean up section headers
        text = re.sub(r'\n([A-Z][^:\n]*:)\n', r'\n\n\1\n', text)  # Space around section headers
        
        # Remove any remaining non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Clean up extra spaces
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on lines
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces on lines
        
        return text.strip()

    def robust_scrape_text(url):
        """Enhanced Beautiful Soup scraping with better error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'noscript', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Extract meaningful content
            content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'div'], 
                                        class_=lambda x: x and any(keyword in x.lower() for keyword in 
                                        ['content', 'article', 'main', 'text', 'body']))
            
            if not content_elements:
                content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            
            text = ' '.join([el.get_text(separator=' ', strip=True) for el in content_elements 
                        if el.get_text(strip=True) and len(el.get_text(strip=True)) > 20])
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = clean_text_for_pdf(text)
            
            return text[:2000] if text else "" 
            
        except Exception as e:
            return f"Error accessing {url}: {str(e)}"

    def try_alternative_sources(region):
        """Try alternative, more accessible sources for regional data"""
        alternative_urls = {
            "EU": [
                "https://www.reuters.com/markets/europe/",
                "https://www.bloomberg.com/europe",
                "https://www.ft.com/european-economy"
            ],
            "NA": [
                "https://www.reuters.com/markets/us/",
                "https://www.bloomberg.com/americas",
                "https://www.marketwatch.com/economy-politics"
            ],
            "AMEA": [
                "https://www.reuters.com/markets/asia/",
                "https://www.bloomberg.com/asia",
                "https://asia.nikkei.com/Economy"
            ],
            "LATAM": [
                "https://www.reuters.com/markets/americas/",
                "https://www.bloomberg.com/latam",
                "https://www.reuters.com/world/americas/"
            ],
            "Global": [
                "https://www.reuters.com/business/",
                "https://www.bloomberg.com/markets",
                "https://www.marketwatch.com/economy-politics"
            ]
        }
        
        urls = alternative_urls.get(region, alternative_urls["Global"])
        scraped_content = []
        
        for url in urls:
            content = robust_scrape_text(url)
            if content and "Error" not in content and len(content) > 100:
                scraped_content.append(content)
                break
        
        return " ".join(scraped_content)

    def generate_with_better_llm_model(region, scraped_data=""):
        """Use better LLM models with enhanced prompting"""
        
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        
        if not OPENROUTER_API_KEY:
            st.error("OpenRouter API key not found in .env file.")
            st.stop()
        
        # Enhanced context with scraped data
        context_data = f"\nReal-time market context: {scraped_data[:1000]}" if scraped_data else ""
        
        # Use only free models to avoid 402 errors
        free_models = [
        "microsoft/wizardlm-2-8x22b:free",
        "google/gemini-flash-1.5:free",
        "mistralai/mistral-7b-instruct:free",
        "meta-llama/llama-3-8b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free"
        ]
        
        enhanced_prompt = f"""
        IMPORTANT: Use only standard ASCII characters in your response. Avoid currency symbols like EUR, GBP, USD symbols - write them as text (EUR, USD, etc.).

        You are a senior manufacturing intelligence analyst with 15+ years experience in {region} PCB manufacturing markets.

        EXPERTISE AREAS:
        - {region} manufacturing cost structures and market dynamics
        - Supply chain optimization and risk management
        - Technology adoption patterns and competitive analysis
        - Regulatory compliance and market positioning strategies

        TASK: Generate a comprehensive manufacturing intelligence report for {region} PCB market.

        CONTEXT:{context_data}

        ANALYSIS FRAMEWORK:
        Use your deep knowledge of {region} manufacturing sector to provide:

        1. **EXECUTIVE SUMMARY** (400+ words):
        - Current {region} market position and size
        - Key trends shaping 2025 market landscape
        - Critical success factors and competitive dynamics
        - Top strategic recommendations with business rationale

        2. **MARKET ANALYSIS** (500+ words):
        - Market size, growth rates, and demand forecasts
        - Competitive landscape and key market players
        - Customer segments and technology adoption trends
        - Regional advantages and positioning opportunities

        3. **OPERATIONAL ASSESSMENT** (400+ words):
        - Manufacturing cost drivers and optimization opportunities
        - Supply chain dynamics and logistics considerations
        - Technology infrastructure and automation levels
        - Quality standards and operational excellence metrics

        4. **RISK ANALYSIS** (400+ words):
        - Political, economic, and regulatory risk factors
        - Supply chain vulnerabilities and market volatility
        - Competitive threats and technology disruption risks
        - Comprehensive mitigation strategies and contingency planning

        5. **STRATEGIC OPPORTUNITIES** (400+ words):
        - Emerging technology applications and growth segments
        - Partnership opportunities and market expansion potential
        - Innovation priorities and competitive differentiation
        - Investment strategies and resource allocation

        6. **RECOMMENDATIONS** (400+ words):
        - Immediate actions (0-6 months) with specific implementation steps
        - Medium-term initiatives (6-18 months) with clear timelines
        - Long-term strategic positioning (18+ months) and vision
        - Success metrics, KPIs, and performance monitoring framework

        REQUIREMENTS:
        - Each section must be comprehensive and exceed minimum word count
        - Include specific percentages, market data, and financial insights
        - Provide concrete examples and actionable recommendations
        - Use professional executive-level business language
        - Focus on {region}-specific market conditions and opportunities
        - IMPORTANT: Use only standard ASCII characters - no special symbols

        Generate each section with detailed, substantive content that provides real business value.
        """

        for model in free_models:
            for attempt in range(2):
                try:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps({
                            "model": model,
                            "messages": [
                                {
                                    "role": "system", 
                                    "content": f"You are a senior manufacturing intelligence analyst specializing in {region} PCB manufacturing markets. Provide comprehensive, detailed business analysis with specific insights and actionable recommendations. Use only standard ASCII characters in your response."
                                },
                                {
                                    "role": "user", 
                                    "content": enhanced_prompt
                                }
                            ],
                            "max_tokens": 4000,
                            "temperature": 0.2,
                            "top_p": 0.9
                        }),
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # Clean the content for PDF compatibility
                        content = clean_text_for_pdf(content)
                        
                        # Parse sections from the response
                        sections = parse_sections_from_response(content)
                        
                        # Clean all sections
                        for key in sections:
                            sections[key] = clean_text_for_pdf(sections[key])
                        
                        if len(sections) >= 6 and all(len(section.split()) > 100 for section in sections.values()):
                            return sections
                    time.sleep(2)
                    
                except Exception as e:
                    time.sleep(2)
        
        st.error("All models failed")
        st.stop()

    def parse_sections_from_response(content):
        """Parse sections from LLM response using keywords"""
        sections = {
            "executive_summary": "",
            "market_analysis": "",
            "operational_assessment": "",
            "risk_analysis": "",
            "strategic_opportunities": "",
            "recommendations": ""
        }
        
        # Split content by section headers
        content_lower = content.lower()
        
        # Find section boundaries
        section_markers = [
            ("executive summary", "executive_summary"),
            ("market analysis", "market_analysis"),
            ("operational assessment", "operational_assessment"),
            ("risk analysis", "risk_analysis"),
            ("strategic opportunities", "strategic_opportunities"),
            ("recommendations", "recommendations")
        ]
        
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            for marker, key in section_markers:
                if marker in line_lower and len(line.strip()) < 100:
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = key
                    current_content = []
                    break
            else:
                # Add content to current section
                if current_section and line.strip():
                    current_content.append(line.strip())
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Fill empty sections with meaningful content
        for key, value in sections.items():
            if not value or len(value.split()) < 50:
                sections[key] = f"Comprehensive {key.replace('_', ' ').title()} analysis for {region_code} PCB manufacturing market. This section provides detailed insights into market conditions, operational factors, and strategic considerations specific to the {region_code} region."
        
        return sections

    # Check cache first
    cache_key = get_pdf_cache_key(region_code, forecast_data)
    cached_pdf_content = load_pdf_content_cache(cache_key)

    if cached_pdf_content:
        # Use cached content dictionary
        pdf_content = cached_pdf_content
    else:
        # Step 1: Try to get some real-time context
        scraped_context = try_alternative_sources(region_code)
        
        # Step 2: Generate with better LLM models
        pdf_content = generate_with_better_llm_model(region_code, scraped_context)

    # NOW create PDF from content (whether cached or newly generated)
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(left=20, top=20, right=20)
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        title_text = clean_text_for_pdf(f'Manufacturing Intelligence Report - {region_code} Region')
        pdf.cell(0, 10, title_text, 0, 1, 'C')
        pdf.ln(5)
        
        # Subtitle
        pdf.set_font('Arial', 'I', 12)
        subtitle_text = clean_text_for_pdf(f'Advanced LLM Analysis | Generated: {datetime.now().strftime("%B %d, %Y")}')
        pdf.cell(0, 8, subtitle_text, 0, 1, 'C')
        pdf.ln(10)
        
        # Add sections to PDF with better formatting
        section_titles = {
            "executive_summary": "Executive Summary",
            "market_analysis": f"{region_code} Market Analysis",
            "operational_assessment": "Operational Assessment",
            "risk_analysis": "Risk Analysis and Mitigation",
            "strategic_opportunities": "Strategic Opportunities",
            "recommendations": "Strategic Recommendations"
        }
        
        for key, title in section_titles.items():
            if key == "market_analysis" or key == "strategic_opportunities":
                pdf.add_page()
            
            # Section title
            pdf.set_font('Arial', 'B', 14 if key in ["executive_summary", "market_analysis"] else 12)
            clean_title = clean_text_for_pdf(title)
            
            # Check if title fits on page
            if len(clean_title) > 80:
                clean_title = clean_title[:77] + "..."
            
            pdf.cell(0, 10 if key in ["executive_summary", "market_analysis"] else 8, clean_title, 0, 1)
            pdf.ln(3)
            
            # Section content with enhanced formatting
            pdf.set_font('Arial', '', 11)
            clean_content = clean_text_for_pdf(pdf_content.get(key, ''))
            
            # Split content into smaller chunks to avoid width issues
            if clean_content:
                # Split by sentences first
                sentences = clean_content.replace('\n\n', ' PARAGRAPH_BREAK ').split('. ')
                
                current_paragraph = ""
                for sentence in sentences:
                    if 'PARAGRAPH_BREAK' in sentence:
                        # End current paragraph and start new one
                        if current_paragraph:
                            try:
                                pdf.multi_cell(0, 6, current_paragraph.strip() + '.')
                                pdf.ln(3)
                            except:
                                # If multi_cell fails, split further
                                words = current_paragraph.split()
                                for i in range(0, len(words), 15):  # 15 words per line
                                    line = ' '.join(words[i:i+15])
                                    pdf.cell(0, 6, line, 0, 1)
                        
                        current_paragraph = sentence.replace('PARAGRAPH_BREAK', '').strip()
                        pdf.ln(3)  # Extra space for new paragraph
                    else:
                        current_paragraph += sentence + '. '
                        
                        # If paragraph gets too long, output it
                        if len(current_paragraph) > 500:
                            try:
                                pdf.multi_cell(0, 6, current_paragraph.strip())
                                pdf.ln(3)
                            except:
                                # If multi_cell fails, split into smaller chunks
                                words = current_paragraph.split()
                                for i in range(0, len(words), 15):
                                    line = ' '.join(words[i:i+15])
                                    pdf.cell(0, 6, line, 0, 1)
                            current_paragraph = ""
                
                # Output remaining content
                if current_paragraph:
                    try:
                        pdf.multi_cell(0, 6, current_paragraph.strip())
                    except:
                        words = current_paragraph.split()
                        for i in range(0, len(words), 15):
                            line = ' '.join(words[i:i+15])
                            pdf.cell(0, 6, line, 0, 1)
            
            pdf.ln(5) 
        
        # Save to cache only if newly generated
        if not cached_pdf_content:
            save_pdf_content_cache(cache_key, pdf_content)
        
        return pdf
        
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        st.stop()

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

def get_detailed_pestel_factor_analysis(factor, factor_name, region):
    """Get detailed LLM analysis for specific PESTEL factor with optimized prompts"""
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    if not OPENROUTER_API_KEY:
        return f"Detailed {factor_name} analysis for {region} PCB manufacturing sector."
    
    # Optimized prompts for each factor
    # prompt = "Say hello."
    prompts = {
        "P": f"""Analyze political factors affecting PCB manufacturing in {region} for 2025-2026.

Focus on:
1. Government policies impacting electronics manufacturing (subsidies, incentives, restrictions)
2. Trade regulations and tariff changes affecting component imports/exports
3. Regulatory compliance requirements (environmental, safety, labor standards)
4. Political stability and upcoming elections that could affect business operations
5. International relations impacting supply chains (US-China tensions, sanctions)

Provide specific examples with quantitative impacts where possible. Limit to 250 words.""",

        "E": f"""Analyze economic conditions in {region} affecting PCB manufacturing costs and demand in 2025.

Cover:
1. Interest rates and inflation impacts on capital investment and raw materials
2. Currency fluctuations affecting import costs (copper, silicon, rare earth elements)
3. Labor cost trends and skilled workforce availability
4. Energy costs and supply security affecting manufacturing operations
5. Economic growth forecasts and electronics demand projections

Include specific percentages, cost ranges, and economic indicators. Limit to 250 words.""",

        "S": f"""Identify social trends in {region} impacting PCB manufacturing and electronics demand through 2025-2026.

Analyze:
1. Demographic shifts affecting workforce availability and consumer electronics demand
2. Environmental consciousness driving sustainable electronics and circular economy
3. Remote work trends changing demand for laptops, networking equipment, IoT devices
4. Education levels and technical skill availability for advanced manufacturing
5. Consumer preferences for locally-made vs. imported electronics

Provide concrete examples and market size implications. Limit to 250 words.""",

        "T": f"""Identify key technologies disrupting PCB manufacturing in {region} over the next 3 years.

Focus on:
1. Advanced manufacturing: AI-driven quality control, automated assembly, predictive maintenance
2. New materials: Flexible PCBs, biodegradable substrates, advanced thermal management
3. Industry 4.0 adoption: IoT sensors, digital twins, smart factory integration
4. Emerging applications: 5G infrastructure, electric vehicles, renewable energy systems
5. Competitive technology gaps and investment priorities

Include adoption timelines and investment requirements. Limit to 250 words.""",

        "En": f"""Analyze environmental factors affecting PCB manufacturing operations in {region}.

Examine:
1. Climate regulations and carbon emission targets affecting manufacturing processes
2. Waste management requirements for electronic waste and hazardous materials
3. Energy efficiency mandates and renewable energy adoption costs
4. Supply chain sustainability pressures from customers and investors
5. Environmental compliance costs (RoHS, REACH, WEEE directives)

Quantify compliance costs and operational changes required. Limit to 250 words.""",

        "L": f"""Summarize legal and regulatory changes in {region} impacting PCB manufacturing in 2025-2026.

Cover:
1. Recent and pending electronics manufacturing regulations and standards
2. Intellectual property laws affecting component sourcing and technology licensing
3. Labor laws and workplace safety requirements for manufacturing facilities
4. Import/export regulations, customs procedures, and certification requirements
5. Data protection laws affecting IoT device manufacturing and testing

Include implementation timelines and compliance costs. Limit to 250 words.""",

        "ML": f"""Provide integrated PESTEL analysis for {region} PCB manufacturing, showing interconnections between factors.

Analyze how:
1. Political trade policies interact with economic currency fluctuations
2. Environmental regulations drive technological innovation requirements
3. Social sustainability demands influence legal compliance frameworks
4. Economic conditions affect political support for manufacturing incentives
5. Technology disruption creates new regulatory and social challenges

Identify the 3 most critical factor combinations affecting profitability. Limit to 300 words."""
    }
    
    #Get the appropriate prompt
    prompt = prompts.get(factor, f"Analyze {factor_name} factors affecting PCB manufacturing in {region}. Provide detailed insights in 250 words.")
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [
                    {"role": "system", "content": f"You are a senior manufacturing analyst specializing in {region} electronics industry. Provide specific, actionable insights with quantitative data where possible."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 400,
                "temperature": 0.3
            }),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content
        else:
            return f"Detailed {factor_name} analysis for {region} PCB manufacturing sector with current market conditions and strategic implications."
            
    except Exception as e:
        return f"Comprehensive {factor_name} assessment for {region} manufacturing operations covering regulatory, market, and operational factors."
    #     if response.status_code == 200:
    #             result = response.json()
    #             content = result["choices"][0]["message"]["content"]
    #             return f"DEBUG SUCCESS: {content}"
    #     else:
    #         return f"DEBUG ERROR: Status {response.status_code} - {response.text}"
            
    # except Exception as e:
    #     return f"DEBUG EXCEPTION: {str(e)}"

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================
st.set_page_config(page_title="Material Intelligence Assistant", layout="wide", initial_sidebar_state="expanded")
col1, col2 = st.columns([10, 1])
with col1:
    st.title("Material Intelligence Assistant")
with col2:
    st.image("bosch.png", width=80)

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

# Initialize PESTEL session state
if 'pestel_results' not in st.session_state:
    st.session_state.pestel_results = None
if 'pestel_displayed' not in st.session_state:
    st.session_state.pestel_displayed = False

# MLPESTEL Analysis Section
st.markdown("---")
st.subheader("MLPESTEL ANALYSIS")

if 'factor_analyses' not in st.session_state:
    st.session_state.factor_analyses = {}

# Regional Analysis Settings
st.subheader("Regional Analysis Settings")

regions = ["Global", "AMEA (Asia-Middle East-Africa)", "EU (European Union)", 
           "NA (North America)", "LATAM (Latin America)"]

if 'selected_region' not in st.session_state:
    st.session_state.selected_region = "Global"

selected_region = st.selectbox(
    "Select Region for All MLPESTEL Analysis:",
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

# PESTEL factor buttons
cols = st.columns(7)
factors = ["ML", "P", "Ec", "S", "T", "En", "L"]
factor_labels = {"ML": "ML", "P": "P", "Ec": "E", "S": "S", "T": "T", "En": "E", "L": "L"}
factor_names = {
    "ML": "Multi-Layer",
    "P": "Political", 
    "Ec": "Economic",
    "S": "Social",
    "T": "Technological",
    "En": "Environmental",
    "L": "Legal"
}

for col, factor in zip(cols, factors):
    if col.button(factor_labels[factor], key=f"mlp_{factor}"):
        current_region = st.session_state.get('selected_region', 'Global')
        cache_key = f"{factor}_{current_region}"
        
        if cache_key not in st.session_state.factor_analyses:
            with st.spinner(f"Analyzing {factor_names[factor]} factor for {current_region}..."):
                analysis = get_detailed_pestel_factor_analysis(factor, factor_names[factor], current_region)
                st.session_state.factor_analyses[cache_key] = analysis
            st.success(f"{factor_names[factor]} analysis completed for {current_region}")     
        else:
            st.info(f"{factor_names[factor]} analysis for {current_region} already displayed below")

# Display all analyses persistently
if st.session_state.factor_analyses:
    st.markdown("---")
    st.subheader("Factor Analysis Results")
    
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear All", type="secondary"):
            st.session_state.factor_analyses = {}
            st.rerun()
    
    for cache_key, analysis in st.session_state.factor_analyses.items():
        parts = cache_key.split('_', 1)
        factor_code = parts[0]
        region_name = parts[1] if len(parts) == 2 else "Unknown Region"
        
        with st.expander(f"{factor_names[factor_code]} Factor Analysis - {region_name}", expanded=True):
            st.markdown(analysis)
            
            if st.button(f"Remove {factor_names[factor_code]}", key=f"remove_{cache_key}"):
                del st.session_state.factor_analyses[cache_key]
                st.rerun()

# Methodology expander
with st.expander("Click to know more about MLPESTEL analysis"):
    st.markdown("""
**Multi-Layer PESTEL (MLPESTEL)** is a next-generation framework that:

1. **Ingests** diverse unstructured data (news, social media) via LLMs.  
2. **Parses** six dimensions: Political, Economic, Social, Technological, Environmental, Legal.  
3. **Scores** each for risk flags (e.g. "high inflation").  
4. **Computes** a base margin, then **adjusts** by weighted risk contributions.  
5. **Outputs** a single **dynamic margin** reflecting current global conditions.

Use MLPESTEL to make your cost & pricing forecasts truly **context-aware**.
    """)

# PESTEL Results and PDF Generation
if 'pestel_results' not in st.session_state:
    st.session_state.pestel_results = None
if 'pestel_displayed' not in st.session_state:
    st.session_state.pestel_displayed = False

if st.button("Apply PESTEL Adjustment to Forecast"):
    try:
        # FIXED: Always use current forecast data from session state
        if 'forecast_results' not in st.session_state or st.session_state.forecast_results is None:
            st.error("No forecast data available. Please generate forecast first.")
            st.stop()
        
        # Get CURRENT forecast data (whether original or regionally adjusted)
        forecast_df = st.session_state.forecast_results['forecast_df']
        
        # Extract ACTUAL costs from current forecast
        if 'manufacturing_cost' in forecast_df.columns:
            base_costs = forecast_df['manufacturing_cost'].tolist()[:3]
        else:
            st.error("No manufacturing_cost column in forecast data")
            st.stop()
        
        current_region = st.session_state.get('selected_region', 'Global')
        
        # FIXED: Use logical PESTEL adjustments that reflect current values
        monthly_price_data = calculate_dynamic_monthly_prices_updated(base_costs, {}, current_region)
        
        # Create PESTEL results using CURRENT forecast values
        results = {
            'forecast_data': {
                'average_cost': sum(base_costs) / len(base_costs),
                'month_1_cost': base_costs[0],
                'month_2_cost': base_costs[1], 
                'month_3_cost': base_costs[2],
                'monthly_prices': monthly_price_data
            },
            'price_analysis': {
                'predicted_selling_price': monthly_price_data[0]['base_price'],
                'adjusted_selling_price': monthly_price_data[0]['adjusted_price'],
                'adjustment_type': 'DYNAMIC',
                'price_change': monthly_price_data[0]['price_change'],
                'final_margin': 22.5
            },
            'margin_data': {'dynamic_margin': 22.5}
        }
        
        st.session_state.pestel_results = results
        st.session_state.pestel_displayed = True
        
        region_status = "with regional constraints" if st.session_state.get('regional_impact_applied', False) else "with original forecast"
        st.success(f"PESTEL analysis completed {region_status} using current forecast values!")
        
    except Exception as e:
        st.error(f"Error in PESTEL analysis: {e}")

# Display PESTEL results
if st.session_state.get('pestel_displayed', False) and st.session_state.get('pestel_results'):
    results = st.session_state.pestel_results
    forecast_data = results['forecast_data']
    price_analysis = results['price_analysis']
    
    # Ensure costs are extracted as numbers
    base_costs = [
        float(forecast_data['month_1_cost']),
        float(forecast_data['month_2_cost']), 
        float(forecast_data['month_3_cost'])
    ]
    
    adj_costs = [float(cost) * 1.08 for cost in base_costs]
    
    if 'monthly_prices' in forecast_data:
        monthly_prices = forecast_data['monthly_prices']
    else:
        current_region = st.session_state.get('selected_region', 'Global')
        monthly_prices = calculate_dynamic_monthly_prices_updated(base_costs, {}, current_region)
    
    months = ['Jan 2025', 'Feb 2025', 'Mar 2025']
    disp_data = {
        "Month": [1, 2, 3],
        "Date": months,
        "Cost (Base)": [f"${cost:,.2f}" for cost in base_costs],
        "Cost (Adj)": [f"${cost:,.2f} ▲" for cost in adj_costs],
        "Price (Base)": [f"${price_data['base_price']:.2f}" for price_data in monthly_prices],
        "Price (Adj)": [format_price_with_arrow(price_data['base_price'], price_data['adjusted_price']) for price_data in monthly_prices]
    }
    
    disp = pd.DataFrame(disp_data)
    
    st.markdown("---")
    st.subheader("Current PESTEL Forecast Results")
    
   # Top row: Table on left, Insights on right
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Adjusted Forecast (Multi-Layer PESTEL Applied)")
        st.dataframe(disp.reset_index(drop=True), use_container_width=True)

    # Add gap between table and insights
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    with col2:
        # Styled box with shadow and better colors
        cost_impact = ((adj_costs[0] / base_costs[0] - 1) * 100)
        
        st.markdown(f"""
        <div style="
            border: 2px solid #1f77b4;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 15px 0;
        ">
            <h4 style="color: #1f77b4; margin-top: 0; text-align: center;"> PESTEL Impact Summary</h4>
            <hr style="border: 1px solid #1f77b4; margin: 10px 0;">
            <p style="margin: 8px 0; font-weight: bold";><strong>Cost Impact:</strong> <span style="color: #9966cc;">{cost_impact:+.1f}%</span> change</p>
            <p style="margin: 8px 0;font-weight: bold";"><strong>Price Adjustment:</strong> <span style="color: #008080;">{price_analysis.get('adjustment_type', 'DYNAMIC')}</span></p>
            <p style="margin: 8px 0;font-weight: bold";"><strong>Final Margin:</strong> <span style="color: #1e4d2b;">{price_analysis.get('final_margin', 22.5):.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)

    
    # Bottom row: Two graphs side by side
    st.subheader("Adjusted Forecast Trends")
    
    graph_col1, graph_col2 = st.columns(2)
    
    base_prices = [float(price_data['base_price']) for price_data in monthly_prices]
    adj_prices = [float(price_data['adjusted_price']) for price_data in monthly_prices]
    month_labels = ['Jan', 'Feb', 'Mar']
    
    with graph_col1:
        # Graph 1: Cost Analysis
        fig1 = go.Figure()
        
        fig1.add_trace(
            go.Scatter(
                x=month_labels, 
                y=base_costs,
                mode='lines+markers', 
                name='Base Cost', 
                line=dict(color='lightblue', width=3),
                marker=dict(size=8)
            )
        )
        
        fig1.add_trace(
            go.Scatter(
                x=month_labels, 
                y=adj_costs,
                mode='lines+markers', 
                name='Adj Cost', 
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            )
        )
        
        fig1.update_layout(
            title='Cost Analysis',
            height=400,
            xaxis=dict(title='Month'),
            yaxis=dict(title='Cost ($)'),
            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
            margin=dict(l=0, r=0, t=30, b=60)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with graph_col2:
        # Graph 2: Price Analysis
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Scatter(
                x=month_labels, 
                y=base_prices,
                mode='lines+markers', 
                name='Base Price', 
                line=dict(color='lightcoral', width=3),
                marker=dict(size=8, symbol='square')
            )
        )
        
        fig2.add_trace(
            go.Scatter(
                x=month_labels, 
                y=adj_prices,
                mode='lines+markers', 
                name='Adj Price', 
                line=dict(color='#ffc2d1', width=3),
                marker=dict(size=8, symbol='square')
            )
        )
        
        fig2.update_layout(
            title='Price Analysis',
            height=400,
            xaxis=dict(title='Month'),
            yaxis=dict(title='Price ($/unit)'),
            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
            margin=dict(l=0, r=0, t=30, b=60)
        )
        
        st.plotly_chart(fig2, use_container_width=True)


    # PESTEL Analysis Insights
    st.markdown("---")
    st.subheader("PESTEL Analysis Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cost Impact", f"{cost_impact:+.1f}%", "Due to PESTEL factors")
    with col2:
        st.metric("Price Adjustment", f"${price_analysis.get('price_change', 0):+.2f}", price_analysis.get('adjustment_type', ''))
    with col3:
        st.metric("Risk Level", "Medium", "Monitor key factors")
    
    # PDF Download Section
    st.markdown("---")
    st.subheader("Export Report")
    
    if st.button("Generate Enhanced PDF Report", key="pdf_download_btn"):
        with st.spinner(f"Generating comprehensive PDF report for {st.session_state.get('selected_region', 'Global')}..."):
            try:
                pdf = generate_enhanced_regional_pdf_report(forecast_data, results)
                pdf_data = pdf.output(dest='S')
                
                if isinstance(pdf_data, (str, bytearray)):
                    pdf_data = bytes(pdf_data, 'latin-1') if isinstance(pdf_data, str) else bytes(pdf_data)
                
                current_region = st.session_state.get('selected_region', 'Global')
                region_code = current_region.split()[0]
                
                st.download_button(
                    label=f"Download {region_code} Report",
                    data=pdf_data,
                    file_name=f"MLPESTEL_Enhanced_{region_code}_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                st.success(f"{region_code} analysis report generated!")
                
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
        
        # Clear PESTEL results button
        if st.button("Clear PESTEL Results", key="clear_pestel"):
            st.session_state.pestel_displayed = False
            st.session_state.pestel_results = None
            st.rerun()


# Regional Impact Application Controls
st.markdown("---")
st.subheader("Regional Forecast Adjustment")

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