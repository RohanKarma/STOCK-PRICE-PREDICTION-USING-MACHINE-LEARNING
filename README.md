# 📈 StockAI Pro - Intelligent Stock Prediction Platform

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![ML](https://img.shields.io/badge/Machine%20Learning-TensorFlow%20%7C%20Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**StockAI Pro** is a comprehensive full-stack machine learning application designed to democratize financial analysis. It bridges the gap between raw market data and actionable insights by combining **Quantitative Analysis** (LSTM/Random Forest predictions) with **Qualitative Analysis** (News Sentiment) in a professional, real-time dashboard.

---

## 🚀 Key Features

### 🧠 **Advanced AI Predictions**
*   **Hybrid Intelligence:** Comparative analysis using **Linear Regression**, **Random Forest**, and **LSTM (Deep Learning)**.
*   **Future Forecasting:** Generates 30-day future price projections with 95% confidence intervals.
*   **High Accuracy:** Achieved **81.9% directional accuracy** on test datasets.

### 📊 **Interactive Visualizations**
*   **Real-Time Charts:** Intraday (1-minute intervals) candlestick charts with auto-refresh.
*   **Technical Indicators:** Overlays for SMA, EMA, Bollinger Bands, and Volume.
*   **Zoom & Pan:** Built with **Plotly** for deep technical analysis.

### 📰 **Live Sentiment Analysis**
*   **News Feed:** Integrates **NewsAPI** to fetch live global financial news.
*   **NLP Engine:** Uses **VADER Sentiment Analysis** to classify headlines as Positive 🟢, Negative 🔴, or Neutral ⚪.
*   **Impact Scoring:** Visualizes sentiment impact on specific assets.

### 🎨 **Professional UI/UX**
*   **Dark Theme:** Custom CSS injection for a professional "Dark Mode" aesthetic.
*   **Glassmorphism:** Modern card-based layout with gradient accents.
*   **Responsive:** Grid layouts that adapt to different screen sizes.

---

## 🛠️ Tech Stack

| Component | Technologies Used |
| **Frontend** | Streamlit, Plotly, Custom CSS, HTML5 |
| **Backend** | Python 3.10, Pandas, NumPy |
| **Machine Learning** | TensorFlow (Keras), Scikit-learn, Joblib |
| **Data Engineering** | yFinance API, NewsAPI, TA-Lib (Technical Analysis) |
| **Database** | SQLite, CSV Caching |

---

## 📂 Project Architecture

The project is structured into three distinct phases:

```text
Stock-Prediction-Project/
├── 📂 phase1_data_pipeline/      # ETL: Fetches and cleans data
│   ├── data_loader.py
│   └── technical_indicators.py
│
├── 📂 phase2_ml_models/          # Intelligence: Trains and saves models
│   ├── models/                   # Saved .h5 and .pkl models
│   ├── train_models.py
│   └── main.py
│
├── 📂 phase3_web_dashboard/      # Presentation: The Streamlit App
│   ├── assets/                   # CSS and Images
│   ├── pages/                    # Multi-page routing
│   ├── utils/                    # Helper functions (Real-time, News)
│   └── app.py                    # Main Entry Point
│
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation


---


## 📸 Project Screenshots

### **1. Dashboard Homepage**
> *The landing page featuring a real-time scrolling ticker, feature cards with glassmorphism UI, and a professional dark theme.*
![Homepage](<img width="940" height="467" alt="image" src="https://github.com/user-attachments/assets/884e7df8-9dd4-4fb0-9acf-dc20b9b237e3" />
)

### **2. AI Predictions & Analysis**
> *Interactive charts showing historical data combined with LSTM model predictions (green line) and 95% confidence intervals.*
![Prediction Chart](screenshots/prediction.png)

### **3. Live News & Sentiment Analysis**
> *Real-time financial news feed where every article is analyzed by AI (VADER) and color-coded: Green (Positive), Red (Negative).*
![News Feed](screenshots/news.png)

### **4. Portfolio Tracker**
> *A grid-based view of a simulated portfolio, displaying current prices, predicted 30-day trends, and risk assessment metrics.*
![Portfolio Tracker](screenshots/portfolio.png)

### **5. Multi-Stock Comparison**
> *Advanced analytical tools including correlation heatmaps to compare performance and volatility across multiple assets.*
![Comparison Tool](screenshots/comparison.png)


---


⚙️ Installation & Setup
Follow these steps to run the project locally.

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/StockAI-Prediction-Platform.git
cd StockAI-Prediction-Platform

2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the Dashboard
Navigate to the dashboard folder and launch the app:

cd phase3_web_dashboard
streamlit run app.py
----
🎮 How to Use
Homepage: View the live ticker and project overview.
Stock Predictions: Select a stock (e.g., AAPL) to view the 30-day forecast and confidence bands.
News Feed: Enter your NewsAPI Key in the sidebar to fetch live news.
Portfolio: Track a simulated portfolio with risk distribution metrics.
Comparison: Select multiple stocks to compare performance and correlation.

🔮 Future Scope
Crypto Support: Extending the pipeline to support Bitcoin and Ethereum.
User Auth: Adding Login/Signup with Firebase for personalized watchlists.
Advanced NLP: Upgrading from VADER to FinBERT for context-aware sentiment analysis.
Alerts: SMS/Email notifications for price breakouts.

⚠️ Disclaimer
This project is for educational purposes only. The predictions generated by the AI models are based on historical patterns and should not be taken as financial advice. Always do your own research before investing.

🤝 Contributing
Contributions are welcome!

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

Developed by Rohan Maurya






