"""HW2 - Multiple Linear Regression | Streamlit Community Cloud app."""
from pathlib import Path
from html import escape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ml_core import analyze_regression, download_public_csv, read_csv_bytes

BASE = Path(__file__).parent
EXAMPLE_URL = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv'

st.set_page_config(page_title='HW2 · 多元線性回歸實驗室', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: #f6f8fb; }
[data-testid="stMetric"] { background: #f6f8fa; padding: 16px; border: 1px solid #e7ebef; border-radius: 12px; }
h1,h2,h3 { letter-spacing: -.025em; }
h1 { color:#14263a; }
.caption-line {color:#65768b;font-size:.95rem;margin-bottom:1rem}
.masthead {border:1px solid #dfe8f2;border-radius:16px;padding:24px 26px;background:#f8fbff;margin-bottom:20px}
.masthead .eyebrow {font-size:.82rem;font-weight:700;letter-spacing:.12em;color:#38689a;text-transform:uppercase}
.masthead h1 {font-size:2rem;margin:.3rem 0;color:#15324e}
.masthead p {margin:0;color:#4b647c}
</style>
''', unsafe_allow_html=True)

st.markdown('''<div class="masthead"><div class="eyebrow">7115056050 · HW2 · CRISP-DM</div>
<h1>多元線性回歸分析實驗室</h1><p>兩種預測問題 · CSV 網址自由輸入 · 特徵選擇 · 模型評估 · 預測區間</p></div>''', unsafe_allow_html=True)

with st.sidebar:
    st.subheader('⚙️ 實驗設定')
    problem = st.selectbox('預測題目（加分：兩種 Problem）',
                           ['① 洪水機率預測（20 個特徵）', '② 新創公司獲利預測'])
    flood = problem.startswith('①')
    source = st.radio('Dataset 來源（加分：可輸入任意公開 CSV）',
                      ['內建示範資料', '公開 CSV 網址', '上傳本機 CSV'], label_visibility='visible')
    uploaded = None
    url = ''
    if source == '公開 CSV 網址':
        url = st.text_input('CSV 網址（需為 HTTPS）', value=EXAMPLE_URL,
                            help='可輸入老師範例的 50_Startups.csv，或其他公開 CSV 連結。')
        st.caption('支援公開 HTTPS CSV，最多 15 MB。私有/內網網址不可使用。')
    elif source == '上傳本機 CSV':
        uploaded = st.file_uploader('選取 CSV 檔案', type=['csv'])
    st.divider()
    st.caption('作業：Multiple Linear Regression + Feature Selection + Model Evaluation')

@st.cache_data(show_spinner=False)
def load_demo(is_flood: bool):
    name = 'flood_sample.csv' if is_flood else '50_Startups.csv'
    return pd.read_csv(BASE / 'data' / name)

@st.cache_data(show_spinner='正在讀取公開 CSV…', ttl=1800)
def load_remote(url: str):
    return read_csv_bytes(download_public_csv(url))

try:
    if source == '內建示範資料':
        df = load_demo(flood)
        origin = 'Kaggle 洪水競賽訓練資料的 16,000 筆固定抽樣' if flood else 'GitHub 公開的 50_Startups.csv'
    elif source == '公開 CSV 網址':
        if not url.strip():
            st.info('請在左側輸入公開 CSV 網址。')
            st.stop()
        df = load_remote(url.strip())
        origin = url.strip()
    else:
        if uploaded is None:
            st.info('請先從左側上傳 CSV。')
            st.stop()
        df = read_csv_bytes(uploaded.getvalue())
        origin = f'上傳資料：{uploaded.name}'
except Exception as exc:
    st.error(f'資料載入失敗：{exc}')
    st.stop()

suggested_target = ('FloodProbability' if flood else 'Profit')
index = df.columns.get_loc(suggested_target) if suggested_target in df.columns else len(df.columns)-1
st.markdown(f'<div class="caption-line">資料來源：{escape(origin[:150])}　｜　原始資料 {len(df):,} 筆 × {df.shape[1]} 欄</div>', unsafe_allow_html=True)
controls = st.columns([2, 2, 2])
with controls[0]:
    target = st.selectbox('預測目標 Target', df.columns.tolist(), index=int(index))
with controls[1]:
    features_raw = max(1, df.shape[1]-1)
    default_k = 15 if flood and source == '內建示範資料' else min(3, features_raw)
    if features_raw == 1:
        k = 1
        st.caption('SelectKBest：目前只有 1 個候選特徵，k = 1。')
    else:
        k = st.slider('SelectKBest 特徵數量 k', min_value=1, max_value=min(30, features_raw), value=min(default_k, min(30, features_raw)))
    st.caption('實際保留數不超過前處理後的有效特徵數，請見模型評估表。')
with controls[2]:
    st.selectbox('模型演算法', ['LinearRegression（OLS）'], disabled=True)

try:
    with st.spinner('模型訓練、特徵選擇與評估中…'):
        result = analyze_regression(df, target=target, k=k)
except Exception as exc:
    st.error(f'無法建立模型：{exc}')
    st.info('請選擇數值型預測目標，並確認 CSV 有至少 25 筆有效資料。')
    st.stop()

met = result.metrics
full_r2 = met.loc['R²', '全部特徵']
selected_label = met.columns[1]
sel_r2 = met.loc['R²', selected_label]
metric_cols = st.columns(4)
metric_cols[0].metric('樣本數', f'{result.sample_rows:,}')
metric_cols[1].metric('模型 A · R²', f'{full_r2:.4f}')
metric_cols[2].metric('模型 B · R²', f'{sel_r2:.4f}')
metric_cols[3].metric('模型 A · RMSE', f'{met.loc["RMSE", "全部特徵"]:.4f}')

if flood and source == '內建示範資料':
    st.caption('📌 網站使用 16,000 筆樣本即時訓練；原作業 Notebook 使用全部 1,117,957 筆，兩者指標可能不同。原 Notebook：模型 A R² = 0.844877；模型 B R² = 0.595773。')

tab_data, tab_model, tab_predict, tab_crisp = st.tabs(['🗂️ 資料理解', '📉 模型評估', '🎯 即時預測', '🧭 CRISP-DM'])

with tab_data:
    st.markdown('### 資料基本資訊')
    cols = st.columns(3)
    cols[0].metric('原始資料筆數', f'{result.source_rows:,}')
    cols[1].metric('有效特徵欄位', str(len(result.input_columns)))
    cols[2].metric('缺失值總數', f'{int(df.isna().sum().sum()):,}')
    st.dataframe(df.head(10), width='stretch')
    with st.expander('展開欄位分類及排除欄位'):
        st.write('數值欄位：', result.numeric_columns)
        st.write('類別欄位：', result.category_columns)
        st.write('排除欄位（ID、單一值或高基數文字）：', result.dropped_columns or '無')
        st.info('資料分割與所有前處理僅使用訓練資料 fit；測試資料只用於最終評估，避免資料洩漏。')
    st.markdown('### 特徵選擇：SelectKBest + f_regression')
    rank = result.feature_ranking.head(15).copy()
    st.dataframe(rank, width='stretch', hide_index=True)
    plot_ranking = rank.dropna(subset=['F 分數']).head(10).iloc[::-1]
    if len(plot_ranking):
        fig, ax = plt.subplots(figsize=(8.8, 4.3))
        ax.barh(plot_ranking['特徵'], plot_ranking['F 分數'], color='#327aa2')
        ax.set_xlabel('F score (higher is stronger univariate association)')
        ax.set_title('Top Feature Scores')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption('F 分數代表單一特徵與目標的關聯程度，不是因果效果，選完特徵也不保證 R² 提高。')

with tab_model:
    st.markdown('### 模型評估：全部特徵 vs SelectKBest')
    st.dataframe(met.style.format('{:.6f}'), width='stretch')
    good_model = '模型 A（全部特徵）' if full_r2 >= sel_r2 else f'模型 B（{len(result.selected_names)} 個特徵）'
    st.success(f'在這次測試集上，R² 較高的是：{good_model}。')
    st.markdown('### 實際值與預測值')
    n = min(700, len(result.y_test))
    rng = np.random.default_rng(42)
    picked = rng.choice(len(result.y_test), n, replace=False)
    actual = result.y_test[picked]
    p1, p2 = result.pred_all[picked], result.pred_selected[picked]
    low = float(min(actual.min(), p1.min(), p2.min()))
    high = float(max(actual.max(), p1.max(), p2.max()))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, pred, name in [(axs[0], p1, 'A · All features'), (axs[1], p2, 'B · Selected features')]:
        ax.scatter(actual, pred, alpha=.37, s=11, color='#2f86aa')
        ax.plot([low, high], [low, high], '--', color='#ba4545', linewidth=1.5)
        ax.set(xlabel='Actual', ylabel='Predicted', title=name)
        ax.grid(alpha=.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('### 模型 A：近似 95% 預測區間')
    idx = rng.choice(len(result.y_test), min(100, len(result.y_test)), replace=False)
    idx = idx[np.argsort(result.pred_all[idx])]
    xaxis = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.fill_between(xaxis, result.interval_lower[idx], result.interval_upper[idx], color='#b2d1e7', alpha=.7, label='Approx. 95% prediction interval')
    ax.plot(xaxis, result.pred_all[idx], color='#2576a3', label='Prediction')
    ax.scatter(xaxis, result.y_test[idx], s=16, color='#df8746', label='Actual')
    ax.set(xlabel='Test samples (sorted by prediction)', ylabel=target,
           title='Approximate 95% Prediction Interval')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f'全部測試集的區間涵蓋率：{result.interval_coverage:.1%}。此為正態近似的單筆觀測預測區間，依賴線性回歸的誤差等假設，不能保證任何一筆預測。')
    st.download_button('下載本次模型指標 CSV', met.to_csv().encode('utf-8-sig'),
                       file_name='hw2_model_metrics.csv', mime='text/csv')

with tab_predict:
    st.markdown('### 自訂一筆輸入資料，立即預測')
    st.caption('數值欄位預設為資料中位數、類別欄位預設為最常見類別。可自行修改。')
    data_row = {}
    with st.form('one_sample'):
        fields = st.columns(3)
        for i, c in enumerate(result.input_columns):
            series = df[c]
            with fields[i % 3]:
                if c in result.numeric_columns:
                    med = float(pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).median())
                    if not np.isfinite(med):
                        med = 0.0
                    data_row[c] = st.number_input(c, value=med, format='%.3f', key=f'num_{c}')
                else:
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if not choices:
                        choices = ['Unknown']
                    data_row[c] = st.selectbox(c, choices, key=f'cat_{c}')
        submitted = st.form_submit_button('開始預測', type='primary', width='stretch')
    if submitted:
        row_df = pd.DataFrame([data_row])
        a, b = result.predict_one(row_df)
        res_cols = st.columns(2)
        res_cols[0].metric('模型 A 預測結果', f'{a:,.4f}')
        res_cols[1].metric('模型 B 預測結果', f'{b:,.4f}')
        if flood and target == 'FloodProbability':
            st.warning('注意：預測值是 Kaggle 資料集中的目標分數，不可當作真實世界的洪水發生機率或預警訊息。')
        st.dataframe(row_df, width='stretch', hide_index=True)

with tab_crisp:
    st.markdown('### CRISP-DM 六階段對照')
    stages = [
        ('1 · Business Understanding', '定義回歸問題：以數值特徵預測連續數值目標，並比較完整特徵與精簡特徵。'),
        ('2 · Data Understanding', '查看資料筆數、欄位、型態、缺失值與特徵的單變數相關性。'),
        ('3 · Data Preparation', '去除 id、填補缺失值、編碼類別變數；80/20 訓練/測試分割。'),
        ('4 · Modeling', '訓練 sklearn LinearRegression，使用 SelectKBest(f_regression) 在訓練集挑選特徵。'),
        ('5 · Evaluation', '比較 MAE、MSE、RMSE、R²；繪製散佈圖與約 95% 預測區間。'),
        ('6 · Deployment', '提供瀏覽器操作介面，選擇題目、輸入資料來源、修改特徵並預測。'),
    ]
    for name, detail in stages:
        with st.expander(name, expanded=name.startswith('1')):
            st.write(detail)
    st.markdown('**本次作業的兩項加分功能**')
    st.write('① 左側下拉選單提供兩種不同的回歸問題。')
    st.write('② 可輸入任意公開 HTTPS CSV 網址（例如老師提供的 50_Startups.csv），並自行選擇目標欄位；也可以上傳自己的 CSV。')
    st.markdown('**參考來源**')
    st.markdown('- [Kaggle Flood Prediction](https://www.kaggle.com/competitions/playground-series-s4e5/data)')
    st.markdown('- [50_Startups.csv（公開 GitHub）](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv)')
    st.markdown('- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)')
    st.markdown('- [scikit-learn SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)')

st.divider()
st.caption('HW2 · 多元線性回歸 / CRISP-DM · 教學演示用途。網站不會把使用者上傳的 CSV 存到 GitHub。')
