# HW2：多元線性回歸分析網站

學號：**7115056050**

- **正式網站：https://7115056050-hw2.streamlit.app/**
- **公開 Repository：https://github.com/juic-12344/7115056050-hw2**
- 部署平台：Streamlit Community Cloud，分支 `main`，入口 `app.py`，Python 3.12。

## 已完成的功能

| 功能 | 網站位置／實作 |
| --- | --- |
| 兩種不同 Problem | 側欄選單：洪水機率預測、新創公司獲利預測 |
| 自訂 Dataset | 內建 CSV、公開 HTTPS CSV 網址、本機 CSV 上傳 |
| 預測目標 | 可選擇數值型 Target 欄位 |
| 資料前處理 | 排除 ID／無效欄位、缺失值填補、類別 One-Hot Encoding |
| 特徵選擇 | `SelectKBest(f_regression)`，可調整 k |
| 模型訓練 | `LinearRegression`，比較全部特徵與選取特徵模型 |
| 模型評估 | MAE、MSE、RMSE、R²；80/20 訓練／測試分割 |
| 圖表 | 特徵 F 分數、實際值對預測值散佈圖、近似 95% 預測區間 |
| 即時預測 | 修改數值及類別欄位，送出一筆自訂資料預測 |
| 結果匯出 | 下載本次模型指標 CSV |
| CRISP-DM | 網站頁籤列出六階段對照 |

## 正式網站驗收（2026-10-22）

已在正式 Streamlit 網站操作確認：兩種 Problem 切換、k=15→5 重新訓練、兩種題目修改輸入後預測、評估表、散佈圖與預測區間、自訂公開 CSV 網址下載與建模、指標 CSV 下載。分享面板顯示 **Make this app public 已勾選**。

另有 **6 項本機自動測試通過**，詳見 [`TEST_RESULTS.md`](TEST_RESULTS.md)。本機環境原先無法解析外部 CSV 主機；正式網站已成功實測下載，該環境限制不影響本次正式部署。

| 內建題目 | 全部特徵 R² | SelectKBest R² |
| --- | ---: | ---: |
| 洪水（k=15） | 0.842665 | 0.590260 |
| 新創公司（k=3） | 0.899311 | 0.916892 |

以上為本專案內建資料的保留測試集結果；更換資料或特徵數後，指標會重新計算。公開來源 CSV 可能與內建版本不同，不宣稱不同來源的指標完全一致。

### 原 Colab 完整資料實驗（維持原始數值）

`analysis/7115056050_hw2.ipynb` 與 PDF 報告保留原本 **1,117,957 筆完整資料**的實驗與已驗證輸出；本次只更新部署說明，沒有重新執行 Notebook 或修改模型數值。

| 原 Colab 模型 | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| 模型 A：20 個特徵 | 0.015792 | 0.000403 | 0.020080 | 0.844877 |
| 模型 B：15 個特徵 | 0.026109 | 0.001051 | 0.032414 | 0.595773 |

網站洪水題目使用 **16,000 筆固定抽樣資料**另外即時訓練，以改善互動速度。上方網站數值與本表完整資料數值屬不同實驗版本，應分開解讀；兩者都保留 20 個原始輸入特徵。Notebook 與 PDF 現已加入正式 GitHub／Streamlit 網址及此版本差異說明。


## 資料來源

1. **洪水機率預測**：Kaggle [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5/data)。內建 `data/flood_sample.csv` 為原始訓練資料的 16,000 筆固定抽樣，包含 20 個原始輸入特徵、id 與 `FloodProbability`。原 Notebook 使用完整 1,117,957 筆資料，指標可能與網站抽樣重訓不同。
2. **新創公司獲利預測**：[50_Startups.csv](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv)。包含 3 個數值特徵、`State` 類別特徵與 `Profit` 目標。此資料作為第二個 Problem 示範；主要 Kaggle 主題為 20 個特徵的洪水資料。

已在正式網站實測的公開 CSV 網址：

```text
https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv
https://github.com/juic-12344/7115056050-hw2/raw/refs/heads/main/data/50_Startups.csv
```

## 操作方式

1. 在側欄選擇 Problem 與 Dataset 來源。
2. 若使用公開 CSV，貼入 HTTPS 原始資料網址；選擇數值型 Target。
3. 調整 k，比較「模型評估」中的兩種模型、指標與圖表。
4. 到「即時預測」修改輸入，再按「開始預測」。

CSV 限制：最大 15 MB、至少 25 筆有效資料、至少一個有效特徵；僅接受公開 HTTPS，拒絕 localhost／私人 IP／內網。高基數文字欄位排除，大型資料最多抽樣 50,000 筆。前處理、欄位篩選及特徵選擇僅依訓練集 fit，測試集用於評估。

## 本機執行與測試

使用 Python 3.12：

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

執行測試：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Windows CMD 可將 `python` 改為 `py`。`requirements.txt` 已鎖定本次驗證的套件版本。

## 專案結構

```text
app.py                  Streamlit 介面
ml_core.py              回歸、特徵選擇與資料載入
requirements.txt        正式部署依賴
requirements-dev.txt    本機測試依賴
README.md               專案與部署說明
TEST_RESULTS.md         本機與正式網站驗收紀錄
test_ml_core.py          核心測試
test_app.py              Streamlit AppTest 測試
data/                   兩份內建 CSV
analysis/               原始 Notebook 與 PDF 報告
```

## 限制與參考

- 洪水輸出為競賽資料的目標分數，不可作為真實災害預警。
- 近似 95% 預測區間依賴線性回歸假設，不保證每筆預測或未來資料都被涵蓋。
- 網站實際演算法為 `LinearRegression`；Ridge／Lasso 若出現在原報告，是文獻比較，並非網站已實作模型。
- 上傳的 CSV 不會提交到 GitHub。私人對話與帳密不放入公開專案。

參考：[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)、[SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)、[Streamlit 部署](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)。
