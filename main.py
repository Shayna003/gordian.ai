import streamlit as st
import json
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import re
import altair as alt
import pandas as pd

# Try to import and initialize Groq with error handling
try:
    from groq import Groq

    def create_groq_client(api_key):
        try:
            return Groq(api_key=api_key)
        except Exception:
            st.error("Groq client initialization failed. Please update the groq package: `pip install --upgrade groq`")
            return None

except ImportError:
    st.error("Groq package not found. Please install: `pip install groq`")
    Groq = None
    create_groq_client = lambda x: None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PackageInfo:
    """Data model for package information"""
    name: str
    version: str
    last_publish: Optional[datetime]
    deprecated: bool
    deprecated_message: Optional[str]
    description: Optional[str]
    dependencies_count: int
    weekly_downloads: int

@dataclass
class AlternativePackage:
    """Data model for alternative package suggestions"""
    name: str
    reason: str
    benefits: List[str]
    migration_difficulty: int
    confidence_score: float

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for API keys and thresholds"""
    # It's recommended to use environment variables for API keys
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

    # Thresholds
    STALE_DAYS_THRESHOLD = 365
    VERY_STALE_DAYS_THRESHOLD = 730

    # API endpoints
    NPM_REGISTRY_BASE = "https://registry.npmjs.org"
    NPM_API_BASE = "https://api.npmjs.org"

    # Models
    GROQ_FAST_MODEL = "llama3-8b-8192"
    GROQ_DEEP_MODEL = "llama3-70b-8192"

# =============================================================================
# NPM PACKAGE ANALYZER
# =============================================================================

class NPMPackageAnalyzer:
    """Analyzes NPM packages for deprecation and staleness"""

    def __init__(self):
        self.session = requests.Session()

    def get_weekly_downloads(self, package_name: str) -> int:
        """Fetch weekly download stats from the npm API"""
        try:
            url = f"{Config.NPM_API_BASE}/downloads/point/last-week/{package_name}"
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json().get("downloads", 0)
            return 0
        except Exception as e:
            logger.error(f"Could not fetch downloads for {package_name}: {e}")
            return 0

    def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Fetch comprehensive package information"""
        try:
            registry_url = f"{Config.NPM_REGISTRY_BASE}/{package_name}"
            response = self.session.get(registry_url)

            if response.status_code != 200:
                return None

            data = response.json()
            latest_version = data.get("dist-tags", {}).get("latest", "")
            version_data = data.get("versions", {}).get(latest_version, {})

            time_data = data.get("time", {})
            last_publish_str = time_data.get(latest_version, time_data.get("modified"))
            last_publish = None
            if last_publish_str:
                try:
                    last_publish = datetime.fromisoformat(last_publish_str.replace('Z', '+00:00'))
                except ValueError:
                    pass

            deprecated = bool(version_data.get("deprecated", False))
            deprecated_message = version_data.get("deprecated") if isinstance(version_data.get("deprecated"), str) else None

            weekly_downloads = self.get_weekly_downloads(package_name)

            return PackageInfo(
                name=package_name,
                version=latest_version,
                last_publish=last_publish,
                deprecated=deprecated,
                deprecated_message=deprecated_message,
                description=data.get("description"),
                dependencies_count=len(version_data.get("dependencies", {})),
                weekly_downloads=weekly_downloads
            )

        except Exception as e:
            logger.error(f"Error fetching package info for {package_name}: {e}")
            return None

    def is_stale(self, package_info: PackageInfo) -> tuple[bool, str]:
        """Determine if a package is stale"""
        if not package_info.last_publish:
            return True, "No publish date available"

        days_since = (datetime.now(package_info.last_publish.tzinfo) - package_info.last_publish).days

        if days_since > Config.VERY_STALE_DAYS_THRESHOLD:
            return True, f"Very stale - last updated {days_since} days ago"
        elif days_since > Config.STALE_DAYS_THRESHOLD:
            return True, f"Stale - last updated {days_since} days ago"

        return False, f"Active - updated {days_since} days ago"

# =============================================================================
# LLM INTEGRATION
# =============================================================================

class LLMAnalyzer:
    """Handles LLM-based package analysis"""

    def __init__(self):
        """Initialize with API key"""
        try:
            self.groq_client = create_groq_client(Config.GROQ_API_KEY)
            if self.groq_client:
                logger.info("GROQ client initialized successfully")
            else:
                logger.warning("GROQ client initialization failed")
        except Exception as e:
            logger.error(f"Failed to initialize GROQ client: {e}")
            self.groq_client = None

    def clean_json_response(self, content: str) -> str:
        """Extracts a JSON object or array from a string."""
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if match:
            return match.group(1).strip()
        
        match = re.search(r'[\{\[].*[\}\]]', content, re.DOTALL)
        if match:
            return match.group(0)
            
        return content

    def analyze_package_with_llm(self, package_info: PackageInfo) -> Dict:
        """Analyze package using GROQ for a detailed dashboard report"""
        if not self.groq_client:
            return {"error": "GROQ client not initialized"}

        try:
            prompt = f"""
Your Role: You are an expert Senior Software Engineer and Security Analyst. Your task is to perform a deep-dive, multi-faceted analysis of a given npm package.

Your Goal: Evaluate the package based on security, maintenance, performance, and popularity. You must synthesize your findings into a single, valid JSON object. Do not include any explanatory text, markdown, or code backticks around the JSON.

Package Information for Analysis:

Name: {package_info.name}

Version: {package_info.version}

Last Published: {package_info.last_publish}

Stated Dependencies: {package_info.dependencies_count}

Weekly Downloads: {package_info.weekly_downloads:,}

Description: {package_info.description}

Analysis & Scoring Criteria (Follow these steps):

Security Analysis (for security_score and vulnerabilities):

Check for known vulnerabilities associated with this package version using the npm audit command philosophy or by checking databases like Snyk or the GitHub Advisory Database.

Examine the package's dependencies. A high number of dependencies, especially with their own vulnerabilities, increases the attack surface.

Assess the package's history of security patches. Frequent security releases are a good sign; old, unpatched vulnerabilities are a major red flag.

Based on these factors, assign a security_score from 0 (critically insecure) to 10 (perfectly secure).
List 3-5 of the most significant vulnerabilities or risks, describing the type of threat (e.g., "Risk of Prototype Pollution due to insecure object merging," "Dependency 'sub-pkg' has a known ReDoS vulnerability"). If no direct CVEs exist, infer potential risks based on the package's function (e.g., "Potential for XSS if used to render user-supplied data without sanitization").

Maintenance Analysis (for maintenance_score):

Review the package's repository (e.g., on GitHub). Look at the date of the last commit and the frequency of recent commits.

Check the open vs. closed issues ratio. A high number of old, unresolved issues is a negative signal.

Consider the last_publish date. A package not published in over a year may be unmaintained.

Based on repository activity, issue resolution, and publish frequency, assign a maintenance_score from 0 (abandoned) to 10 (very actively maintained).

Performance & Bundle Size Analysis (for performance_impact and bundle_size_estimate):

Use a tool like Bundlephobia to determine the minified and gzipped size of the package. Provide this as the bundle_size_estimate string.

Evaluate the computational complexity of the package's primary functions. Does it perform heavy calculations, or is it a simple utility?

Determine the performance_impact:

"low": Small bundle size (< 30KB gzipped) and minimal computational overhead.

"medium": Moderate size or performs operations that could impact performance in tight loops or on large datasets.

"high": Large bundle size (> 150KB gzipped) or performs computationally expensive tasks that are likely to be a performance bottleneck.

Popularity Analysis (for popularity_trend):

Analyze historical download data (e.g., using npmtrends.com). Do not rely solely on the latest weekly_downloads number.

Determine the popularity_trend:

"growing": Download numbers show a clear upward trend over the last 6-12 months.

"stable": Downloads are consistent, with no significant long-term growth or decline.

"declining": Download numbers show a clear downward trend.

Synthesis (for overall_health, detailed_analysis, and alternatives_needed):

Synthesize the security, maintenance, and popularity scores to determine the overall_health: "good," "concerning," or "critical." A low score in any single area (especially security) may warrant a "critical" rating.

Write a detailed_analysis (approximately 150 words) summarizing the package's key strengths (e.g., "lightweight and easy to use for its specific purpose"), weaknesses (e.g., "suffers from a lack of recent updates and has several vulnerable dependencies"), and ideal use cases (e.g., "Best suited for small, internal projects where security risks can be mitigated").

Based on the overall analysis, set the alternatives_needed boolean. Set it to true if the health is "critical" or "concerning," or if major vulnerabilities exist. Otherwise, set it to false.
"""

            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_DEEP_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert npm package analyst. Your response must be a single, valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"LLM JSON parsing failed: {e}\nContent: {content}")
            return {"error": "Failed to parse AI analysis. The response was not valid JSON."}
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return {"error": str(e)}

    def suggest_alternatives(self, package_info: PackageInfo) -> List[AlternativePackage]:
        # This function remains the same
        pass

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(page_title="Gordian.ai", page_icon="🛡️", layout="wide")

    # Inject custom CSS for a cleaner look
    st.markdown("""
        <style>
            .stMetric {
                border: 1px solid #2e2e2e;
                border-radius: 10px;
                padding: 10px;
                background-color: #0F1116;
            }
            .stTabs [data-baseweb="tab-list"] {
        		gap: 24px;
        	}
            .stTabs [data-baseweb="tab"] {
        		height: 50px;
                white-space: pre-wrap;
        		background-color: transparent;
        		border-radius: 4px 4px 0px 0px;
        		gap: 1px;
        		padding-top: 10px;
        		padding-bottom: 10px;
        	}
        	.stTabs [aria-selected="true"] {
        		background-color: #0F1116;
        	}
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Gordian - AI driven npm package analytics")
    st.markdown("##### An AI-powered dashboard to analyze npm package vulnerabilities, performance, and popularity.")

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = NPMPackageAnalyzer()
        st.session_state.llm_analyzer = LLMAnalyzer()

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        if st.session_state.llm_analyzer.groq_client:
            st.success("GROQ API Connected")
        else:
            st.error("GROQ API Not Connected")
        st.info("This app uses Groq for AI analysis. Ensure your API key is set as an environment variable (`GROQ_API_KEY`).")

    # --- Main Layout ---
    col1, col2 = st.columns([1, 3]) # Make the first column smaller

    with col1:
        st.subheader("📦 Analyze a Package")
        package_name = st.text_input("Enter NPM package name", placeholder="e.g., express, react", key="package_input")

        if st.button("🔍 Analyze", type="primary", use_container_width=True) and package_name:
            st.session_state.last_analysis = None # Clear previous analysis
            with st.spinner(f"Fetching data for '{package_name}'..."):
                pkg_info = st.session_state.analyzer.get_package_info(package_name)
                if pkg_info:
                    st.session_state.current_package = pkg_info
                    st.success(f"Loaded **{pkg_info.name}@{pkg_info.version}**")
                    
                    # Display basic info immediately
                    st.markdown(f"**Description**: *{pkg_info.description or 'N/A'}*")
                    if pkg_info.deprecated:
                        st.error(f"**Status**: ⚠️ DEPRECATED")
                        if pkg_info.deprecated_message:
                             st.caption(f"Reason: {pkg_info.deprecated_message}")
                    else:
                        is_stale, reason = st.session_state.analyzer.is_stale(pkg_info)
                        if is_stale:
                            st.warning(f"**Status**: 🕰️ {reason}")
                        else:
                            st.success(f"**Status**: ✅ {reason}")
                else:
                    st.error(f"Package '{package_name}' not found.")
                    st.session_state.current_package = None

    with col2:
        if 'current_package' not in st.session_state or st.session_state.current_package is None:
            st.info("👈 Enter a package name on the left and click 'Analyze' to see the dashboard.")
        else:
            pkg = st.session_state.current_package
            st.header(f"Analytics Dashboard: `{pkg.name}`")

            if st.button("🤖 Run Full AI Analysis", use_container_width=True):
                with st.spinner("🧠 Performing deep analysis with AI... this may take a moment."):
                    analysis = st.session_state.llm_analyzer.analyze_package_with_llm(pkg)
                    st.session_state.last_analysis = analysis
            
            # --- DISPLAY THE DASHBOARD ---
            if 'last_analysis' in st.session_state and st.session_state.last_analysis:
                analysis = st.session_state.last_analysis

                if "error" in analysis:
                    st.error(f"**Analysis Failed**: {analysis['error']}")
                else:
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🛡️ Vulnerabilities & Risks", "🚀 Performance & Dependencies", "📈 Downloads & Popularity"])

                    with tab1:
                        health_emoji = {"good": "🟢", "concerning": "🟡", "critical": "🔴"}.get(analysis.get('overall_health', ''), "⚪")
                        st.metric("Overall Health", f"{health_emoji} {analysis.get('overall_health', 'N/A').title()}")
                        st.markdown("### 📝 AI-Generated Analysis")
                        st.info(analysis.get('detailed_analysis', 'No detailed analysis available.'))
                    
                    with tab2:
                        st.metric("Security Score", f"{analysis.get('security_score', 'N/A')}/10")
                        st.markdown("### ⚠️ Key Vulnerabilities & Risks")
                        risks = analysis.get('vulnerabilities', [])
                        if risks:
                            for risk in risks:
                                st.warning(f"**-** {risk}")
                        else:
                            st.info("No specific vulnerabilities identified by the AI analysis.")

                    with tab3:
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric("Maintenance Score", f"{analysis.get('maintenance_score', 'N/A')}/10")
                        with m_col2:
                            st.metric("Bundle Size Impact", analysis.get('bundle_size_estimate', 'Unknown'))
                        with m_col3:
                            st.metric("Dependencies", f"{pkg.dependencies_count} packages")
                        st.metric("Performance Impact", analysis.get('performance_impact', 'N/A').title())
                        

                    with tab4:
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric("Weekly Downloads", f"{pkg.weekly_downloads:,}")
                        with m_col2:
                            trend_emoji = {"growing": "📈", "stable": "➡️", "declining": "📉"}.get(analysis.get('popularity_trend', ''), "❓")
                            st.metric("Popularity Trend", f"{trend_emoji} {analysis.get('popularity_trend', 'Unknown').title()}")
                        st.markdown(f"*Download stats from npmjs.com, updated weekly.*")


if __name__ == "__main__":
    main()
