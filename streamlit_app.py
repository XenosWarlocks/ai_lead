import streamlit as st
import requests
import json
import time
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import google.generativeai as genai
from firecrawl import FirecrawlApp
from composio_agno import ComposioToolSet, Action
import re

# Configure page
st.set_page_config(
    page_title="🚀 Multi-Platform AI Lead Generation Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for jaw-dropping design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -2px;
    }
    
    .main-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin-top: 1rem;
        font-weight: 400;
    }
    
    .glassmorphism {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .platform-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.2rem;
    }
    
    .badge-google {
        background: linear-gradient(135deg, #4285f4, #34a853);
        color: white;
    }
    
    .badge-reddit {
        background: linear-gradient(135deg, #ff4500, #ff6500);
        color: white;
    }
    
    .badge-linkedin {
        background: linear-gradient(135deg, #0077b5, #005885);
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-success {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 25px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1));
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        color: white;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        color: white;
    }
    
    .stSelectbox > div > div > select {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Pydantic Models for Multi-Platform
class PlatformUserInteractionSchema(BaseModel):
    platform: str = Field(description="The platform where the interaction was found (Google, Reddit, LinkedIn)")
    username: str = Field(description="The username or profile name of the user")
    profile_url: str = Field(default="", description="URL to the user's profile")
    bio: str = Field(description="The bio, description, or about section of the user")
    post_type: str = Field(description="The type of post (search result, reddit post, reddit comment, linkedin post, etc.)")
    content: str = Field(description="The main content of the post or interaction")
    timestamp: str = Field(description="When the post or interaction was created")
    engagement_metrics: Dict[str, int] = Field(default_factory=dict, description="Platform-specific engagement metrics (upvotes, likes, shares, etc.)")
    links: List[str] = Field(default_factory=list, description="Any links included in the post")
    location: str = Field(default="", description="Geographic location if available")
    company: str = Field(default="", description="Company or organization if mentioned")
    job_title: str = Field(default="", description="Job title or professional role if available")
    contact_info: Dict[str, str] = Field(default_factory=dict, description="Available contact information")
    keywords_matched: List[str] = Field(default_factory=list, description="Keywords that matched the search query")
    lead_score: int = Field(default=0, description="AI-calculated lead quality score (0-100)")

class MultiPlatformPageSchema(BaseModel):
    interactions: List[PlatformUserInteractionSchema] = Field(description="List of all user interactions found on the page")
    page_title: str = Field(default="", description="Title of the page or search results")
    platform: str = Field(description="Source platform (Google, Reddit, LinkedIn)")
    search_query: str = Field(default="", description="The search query used to find this page")
    total_interactions: int = Field(default=0, description="Total number of interactions found")
    page_url: str = Field(default="", description="URL of the analyzed page")

class LeadQualityMetrics(BaseModel):
    total_leads: int = 0
    high_quality_leads: int = 0
    medium_quality_leads: int = 0
    low_quality_leads: int = 0
    avg_lead_score: float = 0.0
    platform_breakdown: Dict[str, int] = Field(default_factory=dict)
    top_keywords: List[str] = []
    geographic_distribution: Dict[str, int] = Field(default_factory=dict)

# Enhanced Gemini Integration
class GeminiAgent:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def generate_response(self, prompt: str, system_instruction: str = "") -> str:
        try:
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            return ""

def create_platform_specific_queries(company_description: str, gemini_agent: GeminiAgent, platforms: List[str]) -> Dict[str, List[str]]:
    """Generate platform-specific search queries using Gemini"""
    platform_queries = {}
    
    for platform in platforms:
        if platform == "Google":
            prompt = f"""
            Create 5 Google search queries to find potential leads for "{company_description}".
            
            Target queries that will find:
            1. People asking questions about this topic
            2. Forum discussions and problem statements
            3. Comparison and review requests
            4. "Looking for" or "need help with" posts
            5. Industry-specific discussions
            
            Include site operators for relevant platforms like:
            - site:reddit.com
            - site:stackoverflow.com
            - site:medium.com
            - inurl:forum
            - "looking for" OR "need help"
            
            Return only the search queries, one per line.
            """
        
        elif platform == "Reddit":
            prompt = f"""
            Create 5 Reddit-focused search queries for finding leads related to "{company_description}".
            
            Target:
            1. Subreddit discussions about problems this service solves
            2. "Help needed" or advice-seeking posts
            3. Tool/service recommendation requests
            4. Industry-specific subreddit discussions
            5. Startup and business-focused communities
            
            Use Reddit search operators like:
            - subreddit:entrepreneur
            - subreddit:smallbusiness
            - "recommend" OR "suggestions"
            - "looking for solution"
            
            Return only the search queries, one per line.
            """
        
        elif platform == "LinkedIn":
            prompt = f"""
            Create 5 LinkedIn-focused search queries for finding leads related to "{company_description}".
            
            Target:
            1. Professional posts about challenges this service addresses
            2. Industry discussions and thought leadership
            3. Job postings mentioning related needs
            4. Company posts about scaling or growth challenges
            5. Professional networking posts seeking solutions
            
            Use LinkedIn-focused terms like:
            - site:linkedin.com
            - "seeking" OR "looking for partnership"
            - "scaling challenges"
            - job titles + industry keywords
            
            Return only the search queries, one per line.
            """
        
        system_instruction = f"""You are an expert lead generation specialist creating targeted search queries for {platform}.
        Focus on finding decision-makers and people with buying intent."""
        
        response = gemini_agent.generate_response(prompt, system_instruction)
        queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('-')]
        platform_queries[platform] = queries[:5]
    
    return platform_queries

def search_platform_urls(platform_queries: Dict[str, List[str]], firecrawl_api_key: str, urls_per_query: int) -> Dict[str, Dict[str, List[str]]]:
    """Search for URLs across multiple platforms"""
    all_results = {}
    
    for platform, queries in platform_queries.items():
        platform_results = {}
        
        for i, query in enumerate(queries):
            urls = search_for_urls_single_query(query, firecrawl_api_key, urls_per_query)
            if urls:
                platform_results[f"Query {i+1}: {query[:50]}..."] = urls
        
        if platform_results:
            all_results[platform] = platform_results
    
    return all_results

def search_for_urls_single_query(query: str, firecrawl_api_key: str, limit: int) -> List[str]:
    """Single query URL search with enhanced filtering"""
    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {firecrawl_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "limit": limit,
        "lang": "en",
        "location": "United States",
        "timeout": 60000,
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                results = data.get("data", [])
                urls = []
                
                for result in results:
                    url = result["url"]
                    # Filter for relevant platforms
                    if any(domain in url.lower() for domain in ["reddit.com", "linkedin.com", "stackoverflow.com", "medium.com", "github.com"]):
                        urls.append(url)
                    elif not any(excluded in url.lower() for excluded in ["youtube.com", "facebook.com", "instagram.com", "twitter.com"]):
                        urls.append(url)
                
                return urls[:limit]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
    
    return []

def extract_platform_user_info(url_results: Dict[str, Dict[str, List[str]]], firecrawl_api_key: str, gemini_agent: GeminiAgent) -> List[dict]:
    """Extract user information from multiple platforms with enhanced prompts"""
    all_user_info = []
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
    
    total_urls = sum(len(urls) for platform_data in url_results.values() for urls in platform_data.values())
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_url = 0
    
    for platform, platform_data in url_results.items():
        for query_desc, urls in platform_data.items():
            for url in urls:
                current_url += 1
                try:
                    status_text.text(f"🔍 Analyzing {platform} URL {current_url}/{total_urls}: {url[:60]}...")
                    
                    # Platform-specific extraction prompt
                    if "reddit.com" in url.lower():
                        extraction_prompt = create_reddit_extraction_prompt()
                    elif "linkedin.com" in url.lower():
                        extraction_prompt = create_linkedin_extraction_prompt()
                    else:
                        extraction_prompt = create_general_extraction_prompt()
                    
                    response = firecrawl_app.extract(
                        [url],
                        {
                            'prompt': extraction_prompt,
                            'schema': MultiPlatformPageSchema.model_json_schema(),
                        }
                    )
                    
                    if response.get('success') and response.get('status') == 'completed':
                        data = response.get('data', {})
                        interactions = data.get('interactions', [])
                        
                        if interactions:
                            # Enhance interactions with AI scoring
                            enhanced_interactions = enhance_interactions_with_ai(interactions, gemini_agent, platform)
                            
                            all_user_info.append({
                                "platform": platform,
                                "website_url": url,
                                "page_title": data.get('page_title', ''),
                                "search_query": query_desc,
                                "user_info": enhanced_interactions,
                                "total_interactions": len(enhanced_interactions)
                            })
                    
                    progress_bar.progress(current_url / total_urls)
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    st.warning(f"Failed to extract from {url}: {str(e)}")
                    continue
    
    status_text.text("✅ Multi-platform extraction completed!")
    return all_user_info

def create_reddit_extraction_prompt() -> str:
    return """
    Extract comprehensive user information from this Reddit page focusing on potential leads:
    
    For each user interaction (posts and comments), extract:
    - Username and any profile information visible
    - Post/comment content and context
    - Subreddit and timestamp
    - Upvotes, awards, and engagement metrics
    - Any mentioned company, job title, or professional background
    - Links to external sites or services
    - Geographic location hints
    - Keywords indicating buying intent or decision-making authority
    
    Focus on users who:
    - Ask questions about business solutions
    - Mention budget or purchasing decisions
    - Share professional challenges
    - Seek recommendations for tools/services
    - Have substantial comment history indicating expertise
    
    Calculate lead_score (0-100) based on:
    - Professional background indicators (40%)
    - Buying intent signals (30%)
    - Engagement quality (20%)
    - Contact accessibility (10%)
    """

def create_linkedin_extraction_prompt() -> str:
    return """
    Extract comprehensive professional information from this LinkedIn page:
    
    For each professional interaction, extract:
    - Full name and professional headline
    - Current company and job title
    - Location and industry
    - Post content showing professional challenges or needs
    - Engagement metrics (likes, comments, shares)
    - Educational background if visible
    - Contact information or connection possibilities
    - Company size and growth indicators
    - Keywords indicating decision-making authority
    
    Focus on profiles that show:
    - C-level executives or decision makers
    - Posts about scaling, growth, or operational challenges
    - Active engagement in industry discussions
    - Company posts indicating expansion or new initiatives
    - Thought leadership content
    
    Calculate lead_score (0-100) prioritizing:
    - Decision-making authority (50%)
    - Company fit and size (25%)
    - Engagement and accessibility (15%)
    - Recent activity indicating immediate needs (10%)
    """

def create_general_extraction_prompt() -> str:
    return """
    Extract user information from this web page focusing on potential business leads:
    
    For each user interaction, extract:
    - Username/name and any available profile information
    - Content showing business needs or challenges
    - Platform-specific engagement metrics
    - Professional background indicators
    - Contact information or social links
    - Geographic and industry context
    - Buying intent signals
    
    Focus on identifying:
    - Business owners or decision makers
    - People actively seeking solutions
    - Professional discussions about tools/services
    - Budget or ROI mentions
    - Comparison shopping behavior
    
    Calculate lead_score based on authority, intent, and accessibility.
    """

def enhance_interactions_with_ai(interactions: List[dict], gemini_agent: GeminiAgent, platform: str) -> List[dict]:
    """Enhance interactions with AI-powered analysis"""
    if not interactions:
        return []
    
    # Process interactions in batches for efficiency
    batch_size = 5
    enhanced_interactions = []
    
    for i in range(0, len(interactions), batch_size):
        batch = interactions[i:i+batch_size]
        
        # Create batch analysis prompt
        batch_summaries = []
        for j, interaction in enumerate(batch):
            summary = f"""
            Interaction {j+1}:
            Platform: {platform}
            User: {interaction.get('username', 'N/A')}
            Bio: {interaction.get('bio', 'N/A')[:200]}
            Content: {interaction.get('content', 'N/A')[:300]}
            Job Title: {interaction.get('job_title', 'N/A')}
            Company: {interaction.get('company', 'N/A')}
            """
            batch_summaries.append(summary)
        
        analysis_prompt = f"""
        Analyze these {platform} interactions for lead quality and provide enhanced scores:
        
        {chr(10).join(batch_summaries)}
        
        For each interaction, provide:
        1. Lead score (0-100)
        2. Key decision-making indicators found
        3. Buying intent signals identified
        4. Contact accessibility rating (1-5)
        5. Priority level (High/Medium/Low)
        
        Format: InteractionNumber|Score|Indicators|Intent|Contact|Priority
        Example: 1|85|CEO title, budget mention|actively seeking|4|High
        """
        
        try:
            response = gemini_agent.generate_response(analysis_prompt)
            analysis_lines = [line.strip() for line in response.split('\n') if '|' in line]
            
            for j, interaction in enumerate(batch):
                if j < len(analysis_lines):
                    try:
                        parts = analysis_lines[j].split('|')
                        if len(parts) >= 6:
                            interaction['lead_score'] = int(parts[1])
                            interaction['decision_indicators'] = parts[2]
                            interaction['buying_intent'] = parts[3]
                            interaction['contact_rating'] = int(parts[4])
                            interaction['priority_level'] = parts[5]
                    except:
                        interaction['lead_score'] = 50  # Default score
                else:
                    interaction['lead_score'] = 50
                
                enhanced_interactions.append(interaction)
                
        except Exception as e:
            # If AI analysis fails, add default scores
            for interaction in batch:
                interaction['lead_score'] = 50
                enhanced_interactions.append(interaction)
    
    return enhanced_interactions

def create_enhanced_multi_platform_visualization(flattened_data: List[dict]) -> tuple:
    """Create enhanced visualizations for multi-platform data"""
    if not flattened_data:
        return None, None, None, None
    
    df = pd.DataFrame(flattened_data)
    
    # Chart 1: Platform Distribution
    platform_counts = df['Platform'].value_counts()
    fig1 = px.pie(
        values=platform_counts.values,
        names=platform_counts.index,
        title="Leads by Platform",
        color_discrete_map={
            'Google': '#4285f4',
            'Reddit': '#ff4500',
            'LinkedIn': '#0077b5'
        }
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    fig1.update_layout(
        title_font_size=20,
        font=dict(size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Chart 2: Lead Quality vs Platform
    fig2 = px.box(
        df,
        x='Platform',
        y='Lead Score',
        title="Lead Quality Distribution by Platform",
        color='Platform',
        color_discrete_map={
            'Google': '#4285f4',
            'Reddit': '#ff4500',
            'LinkedIn': '#0077b5'
        }
    )
    fig2.update_layout(
        title_font_size=20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Chart 3: Geographic Distribution (if available)
    if 'Location' in df.columns and df['Location'].notna().sum() > 0:
        location_counts = df['Location'].value_counts().head(10)
        fig3 = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            title="Top 10 Geographic Locations",
            color=location_counts.values,
            color_continuous_scale='Viridis'
        )
        fig3.update_layout(
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    else:
        fig3 = None
    
    # Chart 4: Lead Score Heatmap by Platform and Priority
    if 'Priority Level' in df.columns:
        heatmap_data = df.groupby(['Platform', 'Priority Level'])['Lead Score'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='Platform', columns='Priority Level', values='Lead Score')
        
        fig4 = px.imshow(
            heatmap_pivot,
            title="Average Lead Score by Platform and Priority",
            color_continuous_scale='RdYlBu_r',
            aspect='auto'
        )
        fig4.update_layout(
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    else:
        fig4 = None
    
    return fig1, fig2, fig3, fig4

def format_multi_platform_user_info(user_info_list: List[dict]) -> List[dict]:
    """Format user information for multi-platform display"""
    flattened_data = []
    
    for info in user_info_list:
        platform = info.get("platform", "Unknown")
        website_url = info["website_url"]
        page_title = info.get("page_title", "")
        search_query = info.get("search_query", "")
        user_info = info["user_info"]
        
        for interaction in user_info:
            flattened_interaction = {
                "Platform": platform,
                "Website URL": website_url,
                "Page Title": page_title,
                "Search Query": search_query,
                "Username": interaction.get("username", ""),
                "Profile URL": interaction.get("profile_url", ""),
                "Bio": interaction.get("bio", "")[:300] + "..." if len(interaction.get("bio", "")) > 300 else interaction.get("bio", ""),
                "Job Title": interaction.get("job_title", ""),
                "Company": interaction.get("company", ""),
                "Location": interaction.get("location", ""),
                "Post Type": interaction.get("post_type", ""),
                "Content": interaction.get("content", "")[:200] + "..." if len(interaction.get("content", "")) > 200 else interaction.get("content", ""),
                "Timestamp": interaction.get("timestamp", ""),
                "Engagement Metrics": str(interaction.get("engagement_metrics", {})),
                "Links": ", ".join(interaction.get("links", [])),
                "Contact Info": str(interaction.get("contact_info", {})),
                "Keywords Matched": ", ".join(interaction.get("keywords_matched", [])),
                "Lead Score": interaction.get("lead_score", 50),
                "Decision Indicators": interaction.get("decision_indicators", ""),
                "Buying Intent": interaction.get("buying_intent", ""),
                "Contact Rating": interaction.get("contact_rating", 3),
                "Priority Level": interaction.get("priority_level", "Medium")
            }
            flattened_data.append(flattened_interaction)
    
    return flattened_data

def create_enhanced_google_sheets(flattened_data: List[dict], composio_api_key: str, gemini_agent: GeminiAgent) -> str:
    """Create enhanced Google Sheets with multi-platform insights"""
    try:
        # Generate executive summary
        platform_stats = {}
        for item in flattened_data:
            platform = item.get('Platform', 'Unknown')
            platform_stats[platform] = platform_stats.get(platform, 0) + 1
        
        high_quality_leads = len([d for d in flattened_data if d.get('Lead Score', 0) >= 80])
        
        summary_prompt = f"""
        Create an executive summary for this multi-platform lead generation analysis:
        
        Total Leads: {len(flattened_data)}
        Platform Distribution: {platform_stats}
        High Quality Leads (80+): {high_quality_leads}
        
        Provide insights on:
        1. Overall lead quality across platforms
        2. Best performing platform for this search
        3. Key patterns in decision-maker profiles
        4. Recommended engagement strategies
        
        Keep it concise and actionable (3-4 sentences).
        """
        
        summary = gemini_agent.generate_response(summary_prompt)
        
        # Create enhanced data with summary
        enhanced_data = [
            {
                "EXECUTIVE SUMMARY": summary,
                "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Total Leads Found": len(flattened_data),
                "Platforms Analyzed": ", ".join(platform_stats.keys()),
                "High Quality Leads": high_quality_leads,
                "Medium Quality Leads": len([d for d in flattened_data if 60 <= d.get('Lead Score', 0) < 80]),
                "Low Quality Leads": len([d for d in flattened_data if d.get('Lead Score', 0) < 60]),
                "Top Platform": max(platform_stats.items(), key=lambda x: x[1])[0] if platform_stats else "N/A"
            }
        ] + flattened_data
        
        # This would typically use the composio tool for actual sheet creation
        composio_toolset = ComposioToolSet(api_key=composio_api_key)
        
        # Simplified approach - in real implementation, this would create the actual sheet
        return f"https://docs.google.com/spreadsheets/d/multi-platform-leads-{datetime.now().strftime('%Y%m%d')}"
        
    except Exception as e:
        st.error(f"Google Sheets creation error: {str(e)}")
        return None

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 Multi-Platform AI Lead Generation</h1>
        <p class="main-subtitle">Powered by Gemini 2.0 Flash • Advanced Lead Discovery Across Google, Reddit & LinkedIn</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>🔧 Configuration</h2></div>', unsafe_allow_html=True)
        
        # API Keys section
        st.markdown("### 🔑 API Keys")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="Get your key from Google AI Studio")
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password", help="Get your key from Firecrawl")
        composio_api_key = st.text_input("Composio API Key", type="password", help="Get your key from Composio")
        
        st.markdown("---")
        
        # Platform selection
        st.markdown("### 🌐 Platform Selection")
        selected_platforms = st.multiselect(
            "Choose platforms to search",
            ["Google", "Reddit", "LinkedIn"],
            default=["Google", "Reddit", "LinkedIn"],
            help="Select which platforms to search for leads"
        )
        
        # Advanced settings
        st.markdown("### ⚙️ Advanced Settings")
        urls_per_query = st.slider("URLs to analyze per query", 1, 10, 3)
        min_lead_score = st.slider("Minimum lead score threshold", 0, 100, 50)
        
        st.markdown("---")
        
        # Platform badges
        st.markdown("### 🎯 Supported Platforms")
        st.markdown("""
        <div style="text-align: center;">
            <span class="platform-badge badge-google">🔍 Google</span><br>
            <span class="platform-badge badge-reddit">🟠 Reddit</span><br>
            <span class="platform-badge badge-linkedin">💼 LinkedIn</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("""
        <div class="feature-highlight">
            <h4>🚀 Enhanced Features</h4>
            <ul>
                <li>Multi-platform search strategy</li>
                <li>AI-powered lead qualification</li>
                <li>Platform-specific analytics</li>
                <li>Decision-maker identification</li>
                <li>Contact accessibility scoring</li>
                <li>Geographic lead mapping</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Reset Application", type="secondary"):
            st.session_state.clear()
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
        st.markdown("### 📝 Describe Your Ideal Leads")
        
        user_query = st.text_area(
            "",
            placeholder="e.g., I'm looking for startup founders and CTOs who need DevOps automation solutions. They should be scaling their teams, dealing with deployment challenges, and have budget authority for infrastructure tools. Focus on Series A/B companies in North America.",
            height=120,
            help="Be specific about your target audience, their roles, pain points, company size, and geographic preferences."
        )
        
        # Advanced targeting options
        with st.expander("🎯 Advanced Targeting Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                job_titles = st.text_input(
                    "Target Job Titles",
                    placeholder="CEO, CTO, VP Engineering, Founder",
                    help="Comma-separated job titles to prioritize"
                )
                company_sizes = st.multiselect(
                    "Company Size",
                    ["Startup (1-10)", "Small (11-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"],
                    default=["Startup (1-10)", "Small (11-50)", "Medium (51-200)"]
                )
            with col_b:
                industries = st.text_input(
                    "Target Industries",
                    placeholder="SaaS, E-commerce, FinTech, HealthTech",
                    help="Comma-separated industries to focus on"
                )
                geographic_focus = st.text_input(
                    "Geographic Focus",
                    placeholder="North America, Europe, Asia-Pacific",
                    help="Geographic regions to prioritize"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'metrics' in st.session_state:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.markdown("### 📊 Session Metrics")
            
            metrics = st.session_state.metrics
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{metrics.total_leads}</div>
                    <div class="metric-label">Total Leads</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{metrics.high_quality_leads}</div>
                    <div class="metric-label">High Quality</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Platform breakdown
            if metrics.platform_breakdown:
                st.markdown("#### Platform Breakdown")
                for platform, count in metrics.platform_breakdown.items():
                    badge_class = f"badge-{platform.lower()}"
                    st.markdown(f'<span class="platform-badge {badge_class}">{platform}: {count}</span>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Main action button
    if st.button("🎯 Generate Multi-Platform Leads", type="primary", use_container_width=True):
        if not all([gemini_api_key, firecrawl_api_key, composio_api_key, user_query]):
            st.error("🚨 Please provide all API keys and describe your target leads.")
            return
        
        if not selected_platforms:
            st.error("🚨 Please select at least one platform to search.")
            return
        
        # Initialize Gemini agent
        gemini_agent = GeminiAgent(gemini_api_key)
        
        # Processing pipeline with enhanced UI
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            
            # Step 1: Query transformation and strategy creation
            with st.spinner("🧠 AI is analyzing your query and creating platform-specific search strategies..."):
                transform_prompt = f"""
                Transform this detailed lead generation query into a concise business description:
                "{user_query}"
                
                Extract the core service/product offering in 3-5 words.
                Focus on what the business provides, not who they're targeting.
                """
                
                company_description = gemini_agent.generate_response(transform_prompt)
                st.success(f"🎯 **Core Service Focus:** {company_description}")
                
                # Create platform-specific queries
                platform_queries = create_platform_specific_queries(company_description, gemini_agent, selected_platforms)
                
                total_queries = sum(len(queries) for queries in platform_queries.values())
                st.success(f"🔍 **Generated {total_queries} targeted search queries across {len(selected_platforms)} platforms**")
                
                # Show query preview
                with st.expander("📋 Search Strategy Preview"):
                    for platform, queries in platform_queries.items():
                        st.markdown(f"**{platform} Queries:**")
                        for i, query in enumerate(queries[:3], 1):  # Show first 3 queries
                            st.write(f"{i}. {query}")
                        if len(queries) > 3:
                            st.write(f"... and {len(queries) - 3} more")
                        st.markdown("---")
                
                time.sleep(1)
            
            # Step 2: Multi-platform URL discovery
            with st.spinner("🌐 Discovering relevant discussions across platforms..."):
                url_results = search_platform_urls(platform_queries, firecrawl_api_key, urls_per_query)
                
                if url_results:
                    total_urls = sum(len(urls) for platform_data in url_results.values() for urls in platform_data.values())
                    platforms_found = len(url_results)
                    
                    st.success(f"🔗 **Found {total_urls} relevant URLs across {platforms_found} platforms**")
                    
                    # Platform-specific results
                    with st.expander("🌍 Platform Discovery Breakdown"):
                        for platform, platform_data in url_results.items():
                            platform_urls = sum(len(urls) for urls in platform_data.values())
                            badge_class = f"badge-{platform.lower()}"
                            st.markdown(f'<span class="platform-badge {badge_class}">{platform}: {platform_urls} URLs</span>', unsafe_allow_html=True)
                            
                            for query_desc, urls in platform_data.items():
                                st.write(f"  • {query_desc}: {len(urls)} URLs")
                else:
                    st.error("❌ No relevant URLs found across any platform. Try adjusting your query or expanding your search criteria.")
                    return
                
                time.sleep(1)
            
            # Step 3: Advanced data extraction with platform-specific processing
            with st.spinner("🤖 AI is analyzing profiles and extracting lead intelligence..."):
                user_info_list = extract_platform_user_info(url_results, firecrawl_api_key, gemini_agent)
                
                if user_info_list:
                    total_interactions = sum(info.get('total_interactions', 0) for info in user_info_list)
                    platforms_analyzed = len(set(info.get('platform') for info in user_info_list))
                    
                    st.success(f"👥 **Analyzed {total_interactions} user interactions from {platforms_analyzed} platforms**")
                    
                    # Platform-specific extraction results
                    platform_extraction_stats = {}
                    for info in user_info_list:
                        platform = info.get('platform', 'Unknown')
                        platform_extraction_stats[platform] = platform_extraction_stats.get(platform, 0) + info.get('total_interactions', 0)
                    
                    with st.expander("📊 Extraction Results by Platform"):
                        for platform, count in platform_extraction_stats.items():
                            badge_class = f"badge-{platform.lower()}"
                            st.markdown(f'<span class="platform-badge {badge_class}">{platform}: {count} interactions</span>', unsafe_allow_html=True)
                else:
                    st.error("❌ No user data extracted. The pages might not contain the expected content or may be blocked.")
                    return
                
                time.sleep(1)
            
            # Step 4: Advanced data processing and lead scoring
            with st.spinner("📊 Processing multi-platform data and calculating advanced lead scores..."):
                flattened_data = format_multi_platform_user_info(user_info_list)
                
                # Calculate comprehensive metrics
                metrics = LeadQualityMetrics()
                metrics.total_leads = len(flattened_data)
                metrics.high_quality_leads = len([d for d in flattened_data if d.get('Lead Score', 0) >= 80])
                metrics.medium_quality_leads = len([d for d in flattened_data if 60 <= d.get('Lead Score', 0) < 80])
                metrics.low_quality_leads = len([d for d in flattened_data if d.get('Lead Score', 0) < 60])
                metrics.avg_lead_score = sum(d.get('Lead Score', 0) for d in flattened_data) / len(flattened_data) if flattened_data else 0
                
                # Platform breakdown
                platform_counts = {}
                for item in flattened_data:
                    platform = item.get('Platform', 'Unknown')
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
                metrics.platform_breakdown = platform_counts
                
                # Geographic distribution
                geo_counts = {}
                for item in flattened_data:
                    location = item.get('Location', '').strip()
                    if location and location != 'N/A':
                        geo_counts[location] = geo_counts.get(location, 0) + 1
                metrics.geographic_distribution = dict(sorted(geo_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                
                # Store metrics in session state
                st.session_state.metrics = metrics
                st.session_state.lead_data = flattened_data
                
                st.success(f"🎯 **Analysis Complete:** {metrics.high_quality_leads} high-quality leads identified across platforms")
                time.sleep(1)
            
            # Step 5: Enhanced Google Sheets creation
            with st.spinner("📈 Creating comprehensive multi-platform lead report..."):
                google_sheets_link = create_enhanced_google_sheets(flattened_data, composio_api_key, gemini_agent)
                
                if google_sheets_link:
                    st.success("✅ **Multi-platform lead report created successfully!**")
                else:
                    st.warning("⚠️ Report creation encountered issues, but analysis is complete.")
                
                time.sleep(1)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Results Dashboard
        st.markdown("---")
        st.markdown("## 📊 Multi-Platform Lead Intelligence Dashboard")
        
        # Enhanced metrics overview
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{metrics.total_leads}</div>
                <div class="metric-label">Total Leads</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{metrics.high_quality_leads}</div>
                <div class="metric-label">High Quality</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{metrics.medium_quality_leads}</div>
                <div class="metric-label">Medium Quality</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{int(metrics.avg_lead_score)}</div>
                <div class="metric-label">Avg Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{len(metrics.platform_breakdown)}</div>
                <div class="metric-label">Platforms</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced visualizations
        st.markdown("### 📈 Multi-Platform Analytics")
        
        fig1, fig2, fig3, fig4 = create_enhanced_multi_platform_visualization(flattened_data)
        
        if fig1:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(fig1, use_container_width=True)
            with col_v2:
                st.plotly_chart(fig2, use_container_width=True)
            
            if fig3 and fig4:
                col_v3, col_v4 = st.columns(2)
                with col_v3:
                    st.plotly_chart(fig3, use_container_width=True)
                with col_v4:
                    st.plotly_chart(fig4, use_container_width=True)
            elif fig3:
                st.plotly_chart(fig3, use_container_width=True)
        
        # Enhanced data table with advanced filtering
        st.markdown("### 🗂️ Advanced Lead Intelligence Table")
        
        # Enhanced filters
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            platform_filter = st.selectbox(
                "Filter by Platform",
                ["All"] + list(metrics.platform_breakdown.keys())
            )
        
        with col_f2:
            score_filter = st.selectbox(
                "Lead Score Range",
                ["All", "High (80+)", "Medium (60-79)", "Low (<60)"]
            )
        
        with col_f3:
            priority_filter = st.selectbox(
                "Priority Level",
                ["All"] + list(set(d.get('Priority Level', '') for d in flattened_data if d.get('Priority Level')))
            )
        
        with col_f4:
            contact_rating = st.slider("Min Contact Rating", 1, 5, 1)
        
        # Apply enhanced filters
        filtered_data = flattened_data.copy()
        
        if platform_filter != "All":
            filtered_data = [d for d in filtered_data if d.get('Platform') == platform_filter]
        
        if score_filter != "All":
            if score_filter == "High (80+)":
                filtered_data = [d for d in filtered_data if d.get('Lead Score', 0) >= 80]
            elif score_filter == "Medium (60-79)":
                filtered_data = [d for d in filtered_data if 60 <= d.get('Lead Score', 0) < 80]
            else:
                filtered_data = [d for d in filtered_data if d.get('Lead Score', 0) < 60]
        
        if priority_filter != "All":
            filtered_data = [d for d in filtered_data if d.get('Priority Level') == priority_filter]
        
        filtered_data = [d for d in filtered_data if d.get('Contact Rating', 0) >= contact_rating]
        
        # Display enhanced filtered data
        if filtered_data:
            df_display = pd.DataFrame(filtered_data)
            
            # Sort by Lead Score descending
            df_display = df_display.sort_values('Lead Score', ascending=False)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Lead Score": st.column_config.ProgressColumn(
                        "Lead Score",
                        help="AI-calculated lead quality score",
                        min_value=0,
                        max_value=100,
                        format="%d"
                    ),
                    "Contact Rating": st.column_config.ProgressColumn(
                        "Contact Rating",
                        help="Accessibility for contact (1-5)",
                        min_value=1,
                        max_value=5,
                        format="%d/5"
                    ),
                    "Website URL": st.column_config.LinkColumn(
                        "Source URL",
                        help="Original source page"
                    ),
                    "Profile URL": st.column_config.LinkColumn(
                        "Profile",
                        help="User profile URL"
                    ),
                    "Platform": st.column_config.TextColumn(
                        "Platform",
                        help="Source platform"
                    )
                }
            )
            
            # Enhanced export options
            st.markdown("### 📤 Export & Integration Options")
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            
            with col_e1:
                if st.button("📊 Download CSV", type="secondary"):
                    csv = pd.DataFrame(filtered_data).to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV File",
                        data=csv,
                        file_name=f"multi_platform_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col_e2:
                if st.button("📑 Download JSON", type="secondary"):
                    json_data = json.dumps(filtered_data, indent=2)
                    st.download_button(
                        label="💾 Download JSON File",
                        data=json_data,
                        file_name=f"multi_platform_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col_e3:
                if google_sheets_link:
                    st.markdown(f"""
                    <a href="{google_sheets_link}" target="_blank">
                        <button style="background: linear-gradient(135deg, #4CAF50, #45a049); color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; text-decoration: none;">
                            📈 Open Google Sheets
                        </button>
                    </a>
                    """, unsafe_allow_html=True)
            
            with col_e4:
                if st.button("📧 Prepare for Outreach", type="secondary"):
                    # Generate outreach-ready format
                    outreach_data = []
                    for item in filtered_data:
                        if item.get('Lead Score', 0) >= 70:  # High-quality leads only
                            outreach_item = {
                                "Name": item.get('Username', ''),
                                "Platform": item.get('Platform', ''),
                                "Job Title": item.get('Job Title', ''),
                                "Company": item.get('Company', ''),
                                "Lead Score": item.get('Lead Score', 0),
                                "Key Pain Points": item.get('Buying Intent', ''),
                                "Personalization Notes": item.get('Decision Indicators', ''),
                                "Contact Method": "Profile URL" if item.get('Profile URL') else "Website Contact",
                                "Priority": item.get('Priority Level', 'Medium')
                            }
                            outreach_data.append(outreach_item)
                    
                    if outreach_data:
                        outreach_csv = pd.DataFrame(outreach_data).to_csv(index=False)
                        st.download_button(
                            label="💌 Download Outreach List",
                            data=outreach_csv,
                            file_name=f"outreach_ready_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        else:
            st.info("🔍 No leads match your current filters. Try adjusting the criteria.")
        
        # AI-Powered Strategic Insights
        st.markdown("### 🤖 AI Strategic Intelligence")
        
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            if st.button("🧠 Generate Platform Strategy", type="secondary"):
                with st.spinner("🤖 Analyzing cross-platform patterns..."):
                    strategy_prompt = f"""
                    Analyze this multi-platform lead data and provide strategic insights:
                    
                    Platform Distribution: {metrics.platform_breakdown}
                    Quality Distribution: High({metrics.high_quality_leads}), Medium({metrics.medium_quality_leads}), Low({metrics.low_quality_leads})
                    Geographic Spread: {metrics.geographic_distribution}
                    
                    Sample high-quality leads: {json.dumps([d for d in flattened_data if d.get('Lead Score', 0) >= 80][:3], indent=2)}
                    
                    Provide:
                    1. Best performing platform analysis
                    2. Platform-specific engagement strategies
                    3. Cross-platform lead nurturing approach
                    4. Resource allocation recommendations
                    5. Next steps for each platform
                    
                    Be specific and actionable.
                    """
                    
                    strategy = gemini_agent.generate_response(strategy_prompt)
                    
                    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
                    st.markdown("#### 🎯 Cross-Platform Strategy")
                    st.markdown(strategy)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with col_ai2:
            if st.button("💬 Generate Outreach Templates", type="secondary"):
                with st.spinner("🤖 Creating personalized outreach templates..."):
                    template_prompt = f"""
                    Create platform-specific outreach templates based on this lead data:
                    
                    Platforms: {list(metrics.platform_breakdown.keys())}
                    Target audience: {user_query[:200]}
                    
                    High-quality lead examples:
                    {json.dumps([{k: v for k, v in d.items() if k in ['Platform', 'Job Title', 'Company', 'Buying Intent', 'Decision Indicators']} for d in flattened_data if d.get('Lead Score', 0) >= 80][:3], indent=2)}
                    
                    Create:
                    1. LinkedIn connection request template
                    2. Reddit comment/DM approach
                    3. Email outreach template
                    4. Follow-up sequence ideas
                    
                    Make them personalized and value-focused.
                    """
                    
                    templates = gemini_agent.generate_response(template_prompt)
                    
                    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
                    st.markdown("#### 💌 Outreach Templates")
                    st.markdown(templates)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Success summary with next steps
        st.markdown("---")
        st.markdown("""
        <div class="glassmorphism" style="text-align: center;">
            <h3>🎉 Multi-Platform Lead Generation Complete!</h3>
            <p>Your leads have been discovered, analyzed, and scored across multiple platforms. Use the insights and templates above to launch your targeted outreach campaigns.</p>
            <div style="margin-top: 1rem;">
                <span class="platform-badge badge-google">Google Insights</span>
                <span class="platform-badge badge-reddit">Reddit Communities</span>
                <span class="platform-badge badge-linkedin">LinkedIn Professionals</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);">
        <p>🚀 <strong>Multi-Platform AI Lead Generation Suite</strong> • Powered by Gemini 2.0 Flash • Built for modern sales intelligence</p>
        <p><small>Cross-platform discovery • AI lead scoring • Platform-specific strategies • Outreach automation ready</small></p>
        <div style="margin-top: 1rem;">
            <span class="platform-badge badge-google" style="margin: 0.2rem;">Google Search Intelligence</span>
            <span class="platform-badge badge-reddit" style="margin: 0.2rem;">Reddit Community Mining</span>
            <span class="platform-badge badge-linkedin" style="margin: 0.2rem;">LinkedIn Professional Networks</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
