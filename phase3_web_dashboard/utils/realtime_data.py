"""
Real-time stock data fetcher
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeData:
    """
    Fetch real-time stock data
    """
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
    
    def get_live_price(self):
        """
        Get current live price
        Returns: dict with price info
        """
        try:
            # Get fast info (faster than full info)
            info = self.stock.fast_info
            
            current_price = info.get('lastPrice', None)
            
            # If fast_info doesn't work, try history
            if current_price is None:
                hist = self.stock.history(period='1d', interval='1m')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            # Get more details
            full_info = self.stock.info
            
            return {
                'price': current_price,
                'previous_close': full_info.get('previousClose', current_price),
                'open': full_info.get('open', current_price),
                'day_high': full_info.get('dayHigh', current_price),
                'day_low': full_info.get('dayLow', current_price),
                'volume': full_info.get('volume', 0),
                'market_cap': full_info.get('marketCap', 0),
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Error fetching live price: {e}")
            return None
    
    def get_intraday_data(self, interval='5m', period='1d'):
        """
        Get intraday data for live chart
        
        Args:
            interval: '1m', '5m', '15m', '30m', '1h'
            period: '1d', '5d'
        
        Returns:
            DataFrame with intraday data
        """
        try:
            df = self.stock.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching intraday data: {e}")
            return None
    
    def is_market_open(self):
        """
        Check if market is currently open
        """
        try:
            now = datetime.now()
            
            # Basic check - US market hours (9:30 AM - 4:00 PM EST, Mon-Fri)
            # This is simplified - doesn't account for holidays
            
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Convert to EST (simplified)
            hour = now.hour
            
            if 9 <= hour < 16:  # 9 AM to 4 PM
                if hour == 9 and now.minute < 30:
                    return False
                return True
            
            return False
        
        except:
            return False
    
    def get_live_metrics(self):
        """
        Get comprehensive live metrics
        """
        live_data = self.get_live_price()
        
        if not live_data:
            return None
        
        # Calculate changes
        price = live_data['price']
        prev_close = live_data['previous_close']
        
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        
        return {
            **live_data,
            'change': change,
            'change_pct': change_pct,
            'is_market_open': self.is_market_open()
        }


if __name__ == "__main__":
    # Test
    rtd = RealTimeData('AAPL')
    
    print("\n" + "="*60)
    print("REAL-TIME DATA TEST")
    print("="*60)
    
    metrics = rtd.get_live_metrics()
    
    if metrics:
        print(f"\nTicker: AAPL")
        print(f"Current Price: ${metrics['price']:.2f}")
        print(f"Change: ${metrics['change']:+.2f} ({metrics['change_pct']:+.2f}%)")
        print(f"Day Range: ${metrics['day_low']:.2f} - ${metrics['day_high']:.2f}")
        print(f"Market Status: {'🟢 Open' if metrics['is_market_open'] else '🔴 Closed'}")
        print(f"Last Update: {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")