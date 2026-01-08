"""
Dark Theme Configuration - Aesthetic Color Palette
"""

# Color Palette
DARK_THEME = {
    # ==================
    # BACKGROUNDS
    # ==================
    'bg_primary': '#0a0e27',           # Main background (deep space blue)
    'bg_secondary': '#1a1f3a',         # Cards and containers
    'bg_tertiary': '#252b48',          # Elevated surfaces
    'bg_overlay': 'rgba(26, 31, 58, 0.95)',  # Modal overlays
    
    # ==================
    # GLASS EFFECTS
    # ==================
    'glass_light': 'rgba(255, 255, 255, 0.05)',
    'glass_medium': 'rgba(255, 255, 255, 0.1)',
    'glass_border': 'rgba(255, 255, 255, 0.1)',
    
    # ==================
    # ACCENTS
    # ==================
    'accent_primary': '#667eea',       # Purple-blue
    'accent_secondary': '#f093fb',     # Soft pink
    'accent_tertiary': '#4facfe',      # Cyan blue
    
    # ==================
    # GRADIENTS
    # ==================
    'gradient_primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient_secondary': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'gradient_success': 'linear-gradient(135deg, #4ade80 0%, #10b981 100%)',
    'gradient_info': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'gradient_background': 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%)',
    
    # ==================
    # STATUS COLORS
    # ==================
    'success': '#4ade80',             # Soft green
    'success_dark': '#22c55e',
    'warning': '#fbbf24',             # Soft amber
    'warning_dark': '#f59e0b',
    'error': '#f87171',               # Soft red
    'error_dark': '#ef4444',
    'info': '#60a5fa',                # Soft blue
    'info_dark': '#3b82f6',
    
    # ==================
    # TEXT
    # ==================
    'text_primary': '#e2e8f0',        # Off-white (main text)
    'text_secondary': '#94a3b8',      # Medium gray (secondary text)
    'text_muted': '#64748b',          # Muted gray (disabled/subtle)
    'text_accent': '#a78bfa',         # Purple tint
    
    # ==================
    # BORDERS & DIVIDERS
    # ==================
    'border_light': 'rgba(226, 232, 240, 0.1)',
    'border_medium': 'rgba(226, 232, 240, 0.2)',
    'divider': 'rgba(148, 163, 184, 0.1)',
    
    # ==================
    # SHADOWS
    # ==================
    'shadow_sm': '0 2px 8px rgba(0, 0, 0, 0.3)',
    'shadow_md': '0 4px 16px rgba(0, 0, 0, 0.4)',
    'shadow_lg': '0 8px 32px rgba(0, 0, 0, 0.5)',
    'shadow_glow': '0 0 20px rgba(102, 126, 234, 0.3)',
    
    # ==================
    # CHART COLORS
    # ==================
    'chart_colors': ['#667eea', '#4ade80', '#fbbf24', '#f87171', '#60a5fa', '#a78bfa', '#f093fb'],
}

# CSS for Streamlit
DARK_THEME_CSS = """
<style>
    /* ==================== */
    /* GLOBAL STYLES        */
    /* ==================== */
    
    :root {
        --bg-primary: #0a0e27;
        --bg-secondary: #1a1f3a;
        --bg-tertiary: #252b48;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --accent-primary: #667eea;
        --accent-secondary: #f093fb;
        --success: #4ade80;
        --warning: #fbbf24;
        --error: #f87171;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        color: #e2e8f0;
    }
    
    /* Remove default Streamlit styling */
    .stApp > header {
        background-color: transparent !important;
    }
    
    /* ==================== */
    /* SIDEBAR              */
    /* ==================== */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: transparent;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar widgets */
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stMultiSelect,
    [data-testid="stSidebar"] .stSlider {
        background-color: rgba(37, 43, 72, 0.6);
        border-radius: 8px;
        padding: 8px;
    }
    
    /* ==================== */
    /* TYPOGRAPHY           */
    /* ==================== */
    
    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    p, span, div {
        color: #94a3b8;
    }
    
    /* ==================== */
    /* CARDS & CONTAINERS   */
    /* ==================== */
    
    .element-container,
    .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Card style */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(26, 31, 58, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* ==================== */
    /* METRICS              */
    /* ==================== */
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricDelta"] svg {
        color: #4ade80 !important;
    }
    
    /* ==================== */
    /* BUTTONS              */
    /* ==================== */
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        background: rgba(148, 163, 184, 0.1);
        color: #e2e8f0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: none;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    /* ==================== */
    /* INPUTS & SELECTS     */
    /* ==================== */
    
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div {
        background-color: rgba(37, 43, 72, 0.6) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* ==================== */
    /* DATAFRAMES & TABLES  */
    /* ==================== */
    
    .dataframe {
        background-color: rgba(26, 31, 58, 0.8) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border: none !important;
    }
    
    .dataframe tbody tr {
        background-color: rgba(37, 43, 72, 0.4) !important;
        transition: all 0.2s;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(102, 126, 234, 0.1) !important;
        transform: scale(1.01);
    }
    
    .dataframe tbody tr td {
        color: #e2e8f0 !important;
        padding: 10px !important;
        border-color: rgba(148, 163, 184, 0.1) !important;
    }
    
    /* ==================== */
    /* TABS                 */
    /* ==================== */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(26, 31, 58, 0.6);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
        color: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* ==================== */
    /* EXPANDER             */
    /* ==================== */
    
    .streamlit-expanderHeader {
        background: rgba(26, 31, 58, 0.6) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.1) !important;
        border-color: #667eea !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(37, 43, 72, 0.4) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 0 0 10px 10px !important;
    }
    
    /* ==================== */
    /* SLIDER               */
    /* ==================== */
    
    .stSlider > div > div > div {
        background: rgba(102, 126, 234, 0.2) !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* ==================== */
    /* ALERTS & INFO BOXES  */
    /* ==================== */
    
    .stAlert {
        background-color: rgba(26, 31, 58, 0.8) !important;
        border-radius: 10px !important;
        border-left: 4px solid !important;
        backdrop-filter: blur(10px);
    }
    
    /* Success */
    .stAlert[data-baseweb="notification"] div[role="alert"]:has(svg[data-testid="stSuccessIcon"]) {
        background: rgba(74, 222, 128, 0.1) !important;
        border-left-color: #4ade80 !important;
    }
    
    /* Info */
    .stAlert[data-baseweb="notification"] div[role="alert"]:has(svg[data-testid="stInfoIcon"]) {
        background: rgba(96, 165, 250, 0.1) !important;
        border-left-color: #60a5fa !important;
    }
    
    /* Warning */
    .stAlert[data-baseweb="notification"] div[role="alert"]:has(svg[data-testid="stWarningIcon"]) {
        background: rgba(251, 191, 36, 0.1) !important;
        border-left-color: #fbbf24 !important;
    }
    
    /* Error */
    .stAlert[data-baseweb="notification"] div[role="alert"]:has(svg[data-testid="stErrorIcon"]) {
        background: rgba(248, 113, 113, 0.1) !important;
        border-left-color: #f87171 !important;
    }
    
    /* ==================== */
    /* PROGRESS BAR         */
    /* ==================== */
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #f093fb 100%) !important;
        border-radius: 10px;
    }
    
    /* ==================== */
    /* SPINNER              */
    /* ==================== */
    
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* ==================== */
    /* CUSTOM CLASSES       */
    /* ==================== */
    
    .glass-card {
        background: rgba(26, 31, 58, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    .neon-glow {
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5),
                    0 0 40px rgba(102, 126, 234, 0.3),
                    0 0 60px rgba(102, 126, 234, 0.1);
    }
    
    /* ==================== */
    /* SCROLLBAR            */
    /* ==================== */
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1f3a;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* ==================== */
    /* PLOTLY CHARTS        */
    /* ==================== */
    
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* ==================== */
    /* ANIMATIONS           */
    /* ==================== */
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
        }
        to {
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .slide-in {
        animation: slideIn 0.4s ease-out;
    }
    
    /* ==================== */
    /* RESPONSIVE           */
    /* ==================== */
    
    @media (max-width: 768px) {
        .stButton > button {
            padding: 10px 20px;
            font-size: 14px;
        }
        
        h1 {
            font-size: 28px !important;
        }
        
        [data-testid="stMetric"] {
            padding: 12px;
        }
    }
</style>
"""