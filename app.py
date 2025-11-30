# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime

# ==========================================
# 0. 页面配置 (Page Config)
# ==========================================
st.set_page_config(
    page_title="Robo-Advisor Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==========================================
# 1. 辅助函数 (Helper Functions)
# ==========================================
def categorize_assets(val):
    if val < 100: return "Small (< $100M)"
    if val < 1000: return "Medium ($100M - $1B)"
    if val < 5000: return "Large ($1B - $5B)"
    return "Mega (> $5B)"

def categorize_expense(val):
    if val <= 0.10: return "Very Low (<= 0.10%)"
    if val <= 0.30: return "Low (0.10% - 0.30%)"
    if val <= 0.60: return "Medium (0.30% - 0.60%)"
    return "High (> 0.60%)"

# ==========================================
# 2. 数据处理 (Data Processing)
#    使用 @st.cache_data 缓存数据，避免每次点击按钮都重新加载文件
# ==========================================
@st.cache_data
def load_and_process_data(file_source):
    # file_source 可以是文件路径字符串，也可以是上传的文件对象
    try:
        xls = pd.ExcelFile(file_source)
        price_df = pd.read_excel(xls, sheet_name="Price")
        metadata_df = pd.read_excel(xls, sheet_name="Metadata")
    except Exception as e:
        return None, None, None, f"Error reading file: {e}"

    # --- 1. Clean Metadata ---
    metadata_df.rename(columns={metadata_df.columns[0]: 'Ticker_Full', 'TICKER': 'Ticker'}, inplace=True)
    metadata_df = metadata_df.drop_duplicates(subset=['Ticker_Full'])
    
    num_cols = ['FUND_TOTAL_ASSETS', 'VOLUME_AVG_30D', 'EXPENSE_RATIO', 'BID_ASK_SPREAD', 'TRACKING_ERROR']
    for col in num_cols:
        if col in metadata_df.columns:
            metadata_df[col] = pd.to_numeric(metadata_df[col], errors='coerce')
            metadata_df[col] = metadata_df[col].fillna(metadata_df[col].mean())

    cat_cols = ['FUND_STRATEGY', 'REBALANCING_FREQ']
    for col in cat_cols:
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].fillna('Unknown')

    # History Calculation
    if 'INCEPTION_DATE' in metadata_df.columns:
        metadata_df['INCEPTION_DATE'] = pd.to_datetime(metadata_df['INCEPTION_DATE'], errors='coerce')
        current_year = datetime.now().year
        metadata_df['Years_History'] = current_year - metadata_df['INCEPTION_DATE'].dt.year
        metadata_df['Years_History'] = metadata_df['Years_History'].fillna(0)

    # Categories
    metadata_df['Asset_Category'] = metadata_df['FUND_TOTAL_ASSETS'].apply(categorize_assets)
    metadata_df['Expense_Category'] = metadata_df['EXPENSE_RATIO'].apply(categorize_expense)

    # --- 2. Clean Price ---
    price_df = price_df.drop_duplicates(subset=[price_df.columns[0]])
    price_df.set_index(price_df.columns[0], inplace=True)
    price_data = price_df.T
    price_data.index = pd.to_datetime(price_data.index, errors='coerce')
    price_data.sort_index(inplace=True)
    price_data = price_data.ffill().bfill()
    
    # --- 3. Align Data ---
    common_tickers = list(set(price_data.columns) & set(metadata_df['Ticker_Full']))
    price_data = price_data[common_tickers]
    metadata_df = metadata_df[metadata_df['Ticker_Full'].isin(common_tickers)].copy()
    
    # --- 4. Returns & Stats ---
    log_returns = np.log(price_data / price_data.shift(1))
    log_returns.dropna(inplace=True)
    
    annual_returns = log_returns.mean() * 52
    cov_matrix_annual = log_returns.cov() * 52
    annual_volatility = log_returns.std() * np.sqrt(52)

    metadata_df = metadata_df.set_index('Ticker_Full')
    metadata_df['Annual_Return'] = annual_returns
    metadata_df['Annual_Volatility'] = annual_volatility
    metadata_df = metadata_df.reset_index()

    return metadata_df, log_returns, cov_matrix_annual, None

# ==========================================
# 3. 优化逻辑 (Optimization Logic)
# ==========================================
def get_optimized_portfolio(tickers, mean_returns, cov_matrix, risk_aversion, max_single_weight=1.0):
    num_assets = len(tickers)
    mu = mean_returns[tickers].values
    sigma = cov_matrix.loc[tickers, tickers].values
    
    if np.isnan(mu).any() or np.isnan(sigma).any():
        raise ValueError("Selected ETFs contain NaN data.")

    def objective(weights):
        port_ret = np.sum(mu * weights)
        port_vol_sq = np.dot(weights.T, np.dot(sigma, weights))
        return 0.5 * risk_aversion * port_vol_sq - port_ret

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, max_single_weight) for _ in range(num_assets))
    init_w = np.array([1. / num_assets] * num_assets)
    
    result = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return result.x

# ==========================================
# 4. 主程序界面 (Main App UI)
# ==========================================

# --- Sidebar: Data Loading ---
with st.sidebar:
    st.header("📂 Data Configuration")
    st.write("Upload your ETF Data file (Excel).")
    uploaded_file = st.file_uploader("Upload DATA.xlsx", type=['xlsx'])
    
    # Fallback to local file if no upload
    default_path = r"E:\MFIN\706\term project\DATA.xlsx"
    
    if uploaded_file is not None:
        file_to_load = uploaded_file
        st.success("Using Uploaded File")
    else:
        file_to_load = default_path
        st.info(f"Using Default Path:\n{default_path}")

# Load Data
df_meta, df_returns, df_cov, error = load_and_process_data(file_to_load)

if error:
    st.error(f"Failed to load data. Please check the file path or upload a file.\nError details: {error}")
    st.stop() # Stop execution if no data

# --- Main Page Header ---
st.title("🤖 Intelligent ETF Robo-Advisor")
st.markdown("""
Welcome! This tool helps you build a **customized ETF portfolio** based on your preferences.
**Workflow:** `Filter ETFs` -> `Rank & Select` -> `Optimize with Mean-Variance Model`.
""")
st.markdown("---")

# --- Step 1: Filters ---
st.subheader("1️⃣ Filter Your ETF Universe")

# 布局: 2行3列的筛选器
col1, col2, col3 = st.columns(3)

with col1:
    all_strategies = sorted(list(df_meta['FUND_STRATEGY'].unique()))
    sel_strat = st.multiselect("Fund Strategy", options=all_strategies, default=all_strategies)

with col2:
    asset_order = ["Small (< $100M)", "Medium ($100M - $1B)", "Large ($1B - $5B)", "Mega (> $5B)"]
    avail_assets = [x for x in asset_order if x in df_meta['Asset_Category'].unique()]
    sel_asset = st.multiselect("Asset Size", options=avail_assets, default=avail_assets)

with col3:
    expense_order = ["Very Low (<= 0.10%)", "Low (0.10% - 0.30%)", "Medium (0.30% - 0.60%)", "High (> 0.60%)"]
    avail_expense = [x for x in expense_order if x in df_meta['Expense_Category'].unique()]
    sel_expense = st.multiselect("Expense Ratio", options=avail_expense, default=avail_expense)

col4, col5, col6 = st.columns(3)

with col4:
    all_rebal = sorted(list(df_meta['REBALANCING_FREQ'].unique()))
    sel_rebal = st.multiselect("Rebalancing Frequency", options=all_rebal, default=all_rebal)

with col5:
    max_hist = int(df_meta['Years_History'].max())
    min_yrs = st.slider("Min History (Years)", 0, max_hist, 0)

with col6:
    # 占位符或未来扩展
    st.empty()

# --- Real-time Filter Logic & Counter ---
# Apply Filters
mask = (
    df_meta['FUND_STRATEGY'].isin(sel_strat) &
    df_meta['Asset_Category'].isin(sel_asset) &
    df_meta['Expense_Category'].isin(sel_expense) &
    df_meta['REBALANCING_FREQ'].isin(sel_rebal) &
    (df_meta['Years_History'] >= min_yrs)
)
filtered_df = df_meta[mask].copy()
pool_count = len(filtered_df)

# Display Counter with formatting
st.markdown("##### 📊 Pool Status")
if pool_count < 2:
    st.error(f"🔍 Current Pool: {pool_count} ETFs. (Too few to optimize! Please relax filters.)")
else:
    st.success(f"🔍 Current Pool: {pool_count} ETFs matching criteria.")
    with st.expander("See filtered ETFs list"):
        st.dataframe(filtered_df[['Ticker', 'FUND_STRATEGY', 'Asset_Category', 'EXPENSE_RATIO', 'Annual_Return']])

st.markdown("---")

# --- Step 2: Config & Optimization ---
st.subheader("2️⃣ Configure & Optimize")

c1, c2, c3, c4 = st.columns(4)

with c1:
    sort_opts = ['Highest Liquidity (Volume)', 'Lowest Expense Ratio']
    sel_sort = st.selectbox("Rank Criteria", sort_opts)

with c2:
    top_n = st.slider("Select Top N", 2, 30, 10)

with c3:
    risk_map = {'Aggressive (Low Aversion)': 1.0, 'Balanced (Mid Aversion)': 3.0, 'Conservative (High Aversion)': 10.0}
    sel_risk_label = st.select_slider("Risk Profile", options=list(risk_map.keys()), value='Balanced (Mid Aversion)')
    risk_val = risk_map[sel_risk_label]

with c4:
    max_w = st.slider("Max Weight per ETF", 0.1, 1.0, 0.2, 0.05, format="%.0f%%")

# Run Button
if st.button("🚀 Build Optimized Portfolio", type="primary", use_container_width=True):
    
    if pool_count < 2:
        st.error("Cannot optimize with fewer than 2 ETFs.")
    else:
        try:
            # 1. Ranking
            if sel_sort == 'Highest Liquidity (Volume)':
                filtered_df.sort_values('VOLUME_AVG_30D', ascending=False, inplace=True)
            elif sel_sort == 'Lowest Expense Ratio':
                filtered_df.sort_values('EXPENSE_RATIO', ascending=True, inplace=True)
            
            # 2. Selection
            real_top_n = min(top_n, len(filtered_df))
            
            # Auto-adjust constraints if needed
            if real_top_n * max_w < 1.0:
                st.warning(f"⚠️ Top {real_top_n} ETFs x {max_w:.0%} Max Weight < 100%. Auto-adjusting Max Weight.")
                max_w = 1.0 / real_top_n + 0.05
            
            pool = filtered_df.head(real_top_n).copy()
            tickers = pool['Ticker_Full'].tolist()
            
            # 3. Optimization
            mu_pool = df_meta.set_index('Ticker_Full').loc[tickers, 'Annual_Return']
            weights = get_optimized_portfolio(tickers, mu_pool, df_cov, risk_val, max_single_weight=max_w)
            
            pool['Weight'] = weights
            
            # Stats
            port_ret = np.sum(mu_pool * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(df_cov.loc[tickers, tickers], weights)))
            
            # --- Results Display ---
            st.markdown("### 🏆 Optimization Results")
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Annual Return", f"{port_ret:.2%}")
            m2.metric("Expected Volatility", f"{port_vol:.2%}")
            m3.metric("Selected ETFs", f"{real_top_n}")
            
            # Charts Row
            chart1, chart2 = st.columns([1, 1.5])
            
            with chart1:
                st.markdown("#### Portfolio Allocation")
                w_disp = pool[pool['Weight'] > 0.001]
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                ax1.pie(w_disp['Weight'], labels=w_disp['Ticker'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("husl", len(w_disp)))
                st.pyplot(fig1)

            with chart2:
                st.markdown("#### Efficient Frontier Visualization")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                # Individual ETFs
                sns.scatterplot(x=pool['Annual_Volatility'], y=pool['Annual_Return'], s=100, alpha=0.6, label='Individual ETFs', ax=ax2)
                # Optimized Portfolio
                ax2.scatter(port_vol, port_ret, color='red', s=300, marker='*', label='Optimized Portfolio', zorder=5)
                
                # Annotations
                for i, row in pool.iterrows():
                    ax2.annotate(row['Ticker'], (row['Annual_Volatility'], row['Annual_Return']), fontsize=9)
                
                ax2.set_xlabel('Volatility (Risk)')
                ax2.set_ylabel('Annual Return')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()
                st.pyplot(fig2)

            # Detailed Table
            st.markdown("#### Details")
            
            # Formatting for display
            display_pool = pool[['Ticker', 'FUND_STRATEGY', 'Asset_Category', 'REBALANCING_FREQ', 'Years_History', 'Annual_Return', 'Annual_Volatility', 'Weight']].copy()
            display_pool['Weight'] = display_pool['Weight'].apply(lambda x: f"{x:.2%}")
            display_pool['Annual_Return'] = display_pool['Annual_Return'].apply(lambda x: f"{x:.2%}")
            display_pool['Annual_Volatility'] = display_pool['Annual_Volatility'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_pool, use_container_width=True)

        except Exception as e:
            st.error(f"Optimization Error: {e}")

# Footer
st.markdown("---")
st.caption("© 2025 MFIN 706 Project | ETF Robo-Advisor")

