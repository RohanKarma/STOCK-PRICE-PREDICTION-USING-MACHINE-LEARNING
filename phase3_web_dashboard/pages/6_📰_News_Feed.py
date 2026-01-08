"""
Live News Feed Page - Dark Theme Edition (Complete Fixed)
"""
import streamlit as st
from datetime import datetime
import sys
import os

# Page config
st.set_page_config(page_title="News Feed", page_icon="📰", layout="wide")

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))

# Load dark theme CSS
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'dark_theme.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Import utilities
from news_fetcher import NewsAnalyzer, get_fallback_news
import time

st.title("📰 Live News Feed & Sentiment Analysis")
st.markdown("Real-time market news with AI-powered sentiment analysis")
st.markdown("---")

# Sidebar
st.sidebar.header("🎯 News Settings")

# Stock selection
ticker = st.sidebar.selectbox(
    "Select Stock:",
    ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'All Stocks']
)

# Days back
days_back = st.sidebar.slider("Days Back", 1, 7, 3)

# Max articles
max_articles = st.sidebar.slider("Max Articles", 5, 50, 20)

# Auto-refresh
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Auto-Refresh")

auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

if auto_refresh:
    refresh_rate = st.sidebar.slider("Refresh Rate (minutes)", 5, 60, 15)
    st.sidebar.success(f"✅ Refreshing every {refresh_rate} minutes")

# API Key input
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 NewsAPI Key")

api_key = st.sidebar.text_input(
    "Enter your NewsAPI key:",
    type="password",
    help="Get free key at https://newsapi.org/"
)

if not api_key:
    st.sidebar.warning("⚠️ Enter NewsAPI key for live news")
    st.sidebar.info("👉 Free key: https://newsapi.org/")
else:
    st.sidebar.success("✅ API key configured")

# Initialize analyzer
analyzer = NewsAnalyzer(api_key=api_key if api_key else None)

# Fetch news
st.markdown(f"## 📊 News Analysis for {ticker}")

with st.spinner('Fetching latest news from multiple sources...'):
    if ticker == 'All Stocks':
        # Fetch for all stocks
        all_news = []
        for stock in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']:
            news = analyzer.get_stock_news(stock, days_back=days_back, max_articles=5)
            for article in news:
                article['ticker'] = stock
            all_news.extend(news)
        
        # Sort by date
        news_articles = sorted(all_news, key=lambda x: x['published_at'], reverse=True)[:max_articles]
    else:
        news_articles = analyzer.get_stock_news(ticker, days_back=days_back, max_articles=max_articles)

# Display sentiment summary
if news_articles:
    
    st.success(f"✅ Found {len(news_articles)} recent articles")
    
    st.markdown("### 📊 Sentiment Overview")
    
    summary = analyzer.get_sentiment_summary(news_articles)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_emoji = '🟢' if summary['overall'] == 'Positive' else '🔴' if summary['overall'] == 'Negative' else '⚪'
        st.metric("Overall Sentiment", f"{overall_emoji} {summary['overall']}")
    
    with col2:
        st.metric("Positive News", f"{summary['positive_count']}", f"{summary['positive_pct']:.0f}%")
    
    with col3:
        st.metric("Negative News", f"{summary['negative_count']}", f"{summary['negative_pct']:.0f}%")
    
    with col4:
        st.metric("Neutral News", f"{summary['neutral_count']}", f"{summary['neutral_pct']:.0f}%")
    
    # Sentiment chart - DARK THEME
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Negative', 'Neutral'],
        values=[summary['positive_count'], summary['negative_count'], summary['neutral_count']],
        marker_colors=['#4ade80', '#f87171', '#94a3b8'],
        hole=0.4,
        textfont=dict(color='#ffffff', size=13),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'News Sentiment Distribution',
            'font': {'size': 18, 'color': '#e2e8f0'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.4)',
        font=dict(color='#e2e8f0', size=12),
        height=350,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(102,126,234,0.3)",
            borderwidth=1,
            font=dict(color='#e2e8f0', size=11)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Trending topics
    st.markdown("### 🔥 Trending Topics")
    
    trending = analyzer.get_trending_topics(news_articles, top_n=10)
    
    if trending:
        trend_cols = st.columns(5)
        for i, topic in enumerate(trending):
            with trend_cols[i % 5]:
                st.markdown(f"""
                <div style='
                    background: rgba(102, 126, 234, 0.2);
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid rgba(102, 126, 234, 0.3);
                '>
                    <div style='color: #60a5fa; font-weight: bold; font-size: 14px;'>{topic['topic'].capitalize()}</div>
                    <div style='color: #94a3b8; font-size: 12px;'>{topic['count']} mentions</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📊 No trending topics found")
    
    st.markdown("---")
    
    # Display news articles - STREAMLIT NATIVE VERSION (NO HTML ISSUES)
    st.markdown(f"### 📰 Latest Articles")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment:",
            ['Positive', 'Negative', 'Neutral'],
            default=['Positive', 'Negative', 'Neutral'],
            key='sentiment_filter_main'
        )
    
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by:",
            ['Latest First', 'Oldest First', 'Most Positive', 'Most Negative'],
            key='sort_by_main'
        )
    
    # Filter articles
    filtered_articles = [a for a in news_articles if a['sentiment'] in sentiment_filter]
    
    # Sort articles
    if sort_by == 'Latest First':
        filtered_articles.sort(key=lambda x: x['published_at'], reverse=True)
    elif sort_by == 'Oldest First':
        filtered_articles.sort(key=lambda x: x['published_at'])
    elif sort_by == 'Most Positive':
        filtered_articles.sort(key=lambda x: x['sentiment_compound'], reverse=True)
    elif sort_by == 'Most Negative':
        filtered_articles.sort(key=lambda x: x['sentiment_compound'])
    
    if filtered_articles:
        st.info(f"📄 Showing {len(filtered_articles)} articles")
        
        # Display articles using STREAMLIT COMPONENTS (No HTML rendering issues)
        for i, article in enumerate(filtered_articles):
            
            # Determine sentiment styling
            if article['sentiment'] == 'Positive':
                sentiment_emoji = '🟢'
                sentiment_color = '#4ade80'
            elif article['sentiment'] == 'Negative':
                sentiment_emoji = '🔴'
                sentiment_color = '#f87171'
            else:
                sentiment_emoji = '⚪'
                sentiment_color = '#94a3b8'
            
            # Format time
            try:
                pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                time_ago = datetime.now() - pub_date.replace(tzinfo=None)
                
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"
            except:
                time_str = "Recently"
            
            # Create container for each article
            with st.container():
                # Header with sentiment badge
                header_col1, header_col2 = st.columns([3, 1])
                
                with header_col1:
                    st.markdown(f"## {sentiment_emoji} {article['title']}")
                
                with header_col2:
                    st.markdown(f"<div style='text-align: right; color: #94a3b8;'>🕐 {time_str}</div>", unsafe_allow_html=True)
                
                # Metadata row
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                
                with meta_col1:
                    st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}; font-weight: bold;'>{article['sentiment']}</span>", unsafe_allow_html=True)
                
                with meta_col2:
                    st.markdown(f"**Source:** {article['source']}")
                
                with meta_col3:
                    if ticker == 'All Stocks' and 'ticker' in article:
                        st.markdown(f"**Stock:** {article['ticker']}")
                
                # Description
                if article.get('description') and article['description'] != 'No description':
                    st.write(article['description'])
                
                # Sentiment score
                score_col1, score_col2 = st.columns([4, 1])
                
                with score_col1:
                    score_normalized = (article['sentiment_score'] + 1) / 2
                    st.progress(score_normalized)
                
                with score_col2:
                    st.metric("Score", f"{article['sentiment_score']:.2f}")
                
                # Read more button
                if article.get('url') and article['url'] != '#':
                    st.link_button("🔗 Read Full Article", article['url'])
                
                st.markdown("---")
    
    else:
        st.warning("📭 No articles match your filters.")

else:
    st.warning("📭 No news articles found.")
    
    if not api_key:
        st.error("❌ **No API Key:** Please enter your NewsAPI key in the sidebar")
        
        with st.expander("🔑 How to get NewsAPI Key"):
            st.markdown("""
            **Steps to get your free API key:**
            
            1. Go to https://newsapi.org/
            2. Click "Get API Key" or "Register"
            3. Fill in your details (email, name)
            4. Confirm your email
            5. Copy your API key
            6. Paste it in the sidebar input field
            
            **Free Tier Includes:**
            - 100 requests per day
            - Access to 80,000+ news sources
            - Up to 1 month of historical news
            """)
    else:
        st.info("💡 Try adjusting the filters or check back later.")
    
    # Show sample data
    st.markdown("---")
    st.markdown("### 📰 Sample News (Demo Data)")
    st.info("Enter your API key to see real news.")
    
    sample = get_fallback_news(ticker if ticker != 'All Stocks' else 'AAPL')
    
    for article in sample[:5]:
        
        # Determine colors
        if article['sentiment'] == 'Positive':
            emoji = '🟢'
            color = '#4ade80'
        elif article['sentiment'] == 'Negative':
            emoji = '🔴'
            color = '#f87171'
        else:
            emoji = '⚪'
            color = '#94a3b8'
        
        with st.container():
            st.markdown(f"### {emoji} {article['title']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Sentiment:** <span style='color: {color};'>{article['sentiment']}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Source:** {article['source']}")
            
            st.write(article['description'])
            st.markdown("---")

# Auto-refresh
if auto_refresh and api_key:
    time.sleep(refresh_rate * 60)
    st.rerun()

# Footer
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("📰 News powered by NewsAPI.org")

with footer_col2:
    st.caption("🤖 Sentiment analysis by VADER")

with footer_col3:
    if api_key:
        st.caption("✅ Live data active")
    else:
        st.caption("⚠️ Demo mode - Add API key")