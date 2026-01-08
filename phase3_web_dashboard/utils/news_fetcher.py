"""
Real-time news fetcher and sentiment analyzer
"""
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """
    Fetch and analyze stock news
    """
    
    def __init__(self, api_key=None):
        """
        Args:
            api_key: NewsAPI key (get from https://newsapi.org/)
        """
        # Default API key (replace with yours!)
        self.api_key = api_key or "fb21cbcafc6e44ea8f6d5cdd19c9bd2a"
        self.base_url = "https://newsapi.org/v2/everything"
        self.vader = SentimentIntensityAnalyzer()
    
    def get_stock_news(self, ticker, company_name=None, days_back=7, max_articles=20):
        """
        Fetch news for a specific stock
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            company_name: Full company name (e.g., 'Apple Inc')
            days_back: Number of days to look back
            max_articles: Maximum number of articles
        
        Returns:
            List of news articles with sentiment
        """
        try:
            # Company name mapping
            company_names = {
                'AAPL': 'Apple Inc',
                'GOOGL': 'Google Alphabet',
                'MSFT': 'Microsoft',
                'TSLA': 'Tesla',
                'AMZN': 'Amazon'
            }
            
            if not company_name:
                company_name = company_names.get(ticker, ticker)
            
            # Build search query
            query = f'("{ticker}" OR "{company_name}") AND (stock OR shares OR market)'
            
            # Date range
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # API parameters
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': max_articles,
                'apiKey': self.api_key
            }
            
            # Make request
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"NewsAPI error: {response.status_code}")
                return []
            
            data = response.json()
            
            if data['status'] != 'ok':
                logger.error(f"NewsAPI returned error: {data.get('message')}")
                return []
            
            articles = data.get('articles', [])
            
            # Process and add sentiment
            processed_articles = []
            
            for article in articles:
                # Skip if no title or description
                if not article.get('title') or article.get('title') == '[Removed]':
                    continue
                
                # Analyze sentiment
                sentiment = self._analyze_sentiment(
                    article.get('title', '') + ' ' + article.get('description', '')
                )
                
                processed_article = {
                    'title': article.get('title', 'No title'),
                    'description': article.get('description', 'No description'),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'image': article.get('urlToImage', ''),
                    'sentiment': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                    'sentiment_compound': sentiment['compound']
                }
                
                processed_articles.append(processed_article)
            
            logger.info(f"Fetched {len(processed_articles)} articles for {ticker}")
            
            return processed_articles
        
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER
        
        Returns:
            dict with sentiment label and scores
        """
        try:
            # VADER sentiment (better for social media/news)
            vader_scores = self.vader.polarity_scores(text)
            compound = vader_scores['compound']
            
            # Classify sentiment
            if compound >= 0.05:
                label = 'Positive'
                emoji = '🟢'
            elif compound <= -0.05:
                label = 'Negative'
                emoji = '🔴'
            else:
                label = 'Neutral'
                emoji = '⚪'
            
            return {
                'label': label,
                'emoji': emoji,
                'score': abs(compound),
                'compound': compound,
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu']
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                'label': 'Neutral',
                'emoji': '⚪',
                'score': 0,
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 1
            }
    
    def get_sentiment_summary(self, articles):
        """
        Get overall sentiment summary from articles
        
        Args:
            articles: List of articles with sentiment
        
        Returns:
            dict with sentiment summary
        """
        if not articles:
            return {
                'overall': 'Neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'avg_sentiment': 0
            }
        
        sentiments = [a['sentiment'] for a in articles]
        compounds = [a['sentiment_compound'] for a in articles]
        
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        neutral_count = sentiments.count('Neutral')
        
        avg_compound = sum(compounds) / len(compounds)
        
        # Overall sentiment
        if avg_compound >= 0.05:
            overall = 'Positive'
        elif avg_compound <= -0.05:
            overall = 'Negative'
        else:
            overall = 'Neutral'
        
        return {
            'overall': overall,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'avg_sentiment': avg_compound,
            'positive_pct': (positive_count / len(articles)) * 100,
            'negative_pct': (negative_count / len(articles)) * 100,
            'neutral_pct': (neutral_count / len(articles)) * 100
        }
    
    def get_trending_topics(self, articles, top_n=5):
        """
        Extract trending topics from articles
        
        Args:
            articles: List of articles
            top_n: Number of top topics to return
        
        Returns:
            List of trending topics
        """
        # Simple keyword extraction (can be enhanced with NLP)
        keywords = {}
        
        for article in articles:
            text = article.get('title', '') + ' ' + article.get('description', '')
            words = text.lower().split()
            
            # Filter common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                           'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                           'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                           'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            for word in words:
                word = word.strip('.,!?;:"\'()[]{}').lower()
                if len(word) > 3 and word not in common_words:
                    keywords[word] = keywords.get(word, 0) + 1
        
        # Sort by frequency
        trending = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [{'topic': word, 'count': count} for word, count in trending]


# Fallback news data generator (when API fails or limit reached)
def get_fallback_news(ticker):
    """
    Generate sample news when API is unavailable
    """
    sample_news = [
        {
            'title': f'{ticker} Shows Strong Market Performance',
            'description': 'Analysts predict continued growth based on recent market trends and technical indicators.',
            'url': '#',
            'source': 'Market Watch',
            'published_at': datetime.now().isoformat(),
            'image': None,
            'sentiment': 'Positive',
            'sentiment_score': 0.7,
            'sentiment_compound': 0.7
        },
        {
            'title': f'{ticker} Stock Analysis: What Investors Need to Know',
            'description': 'Market experts weigh in on the latest developments and future outlook for the company.',
            'url': '#',
            'source': 'Financial Times',
            'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
            'image': None,
            'sentiment': 'Neutral',
            'sentiment_score': 0.1,
            'sentiment_compound': 0.1
        }
    ]
    
    return sample_news


if __name__ == "__main__":
    # Test
    analyzer = NewsAnalyzer()
    
    print("\n" + "="*60)
    print("NEWS ANALYZER TEST")
    print("="*60)
    
    # Fetch news for AAPL
    news = analyzer.get_stock_news('AAPL', max_articles=5)
    
    if news:
        print(f"\nFetched {len(news)} articles for AAPL\n")
        
        for i, article in enumerate(news[:3], 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Sentiment: {article['sentiment']} (Score: {article['sentiment_score']:.2f})")
            print(f"   Published: {article['published_at']}")
            print()
        
        # Sentiment summary
        summary = analyzer.get_sentiment_summary(news)
        print("\nSentiment Summary:")
        print(f"Overall: {summary['overall']}")
        print(f"Positive: {summary['positive_count']} ({summary['positive_pct']:.1f}%)")
        print(f"Negative: {summary['negative_count']} ({summary['negative_pct']:.1f}%)")
        print(f"Neutral: {summary['neutral_count']} ({summary['neutral_pct']:.1f}%)")
    else:
        print("\n⚠️ No news fetched. Please check your API key.")
        print("Get a free API key at: https://newsapi.org/")