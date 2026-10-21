# HW2：多元線性回歸分析網站（7115056050）

本專案依照 **更新版 HW2 規定**製作：主要繳交為 **GitHub 儲存庫與可瀏覽的網頁 URL**，保留 CRISP-DM 六階段、特徵選擇（SelectKBest）與模型評估，另外新增兩項加分功能。

## 部署狀態與驗證（2026-10-22）

- 程式、資料集與依賴已檢查；Python 3.12.14 本機自動測試 **6 項通過**。
- 已修正單一特徵 CSV 的滑桿錯誤、無限值處理；欄位篩選以訓練集為依據，並更新 Streamlit 元件參數。
- GitHub Repository 與 Streamlit Community Cloud 正式部署：**等待帳號登入／授權，尚未完成**。
- 公開 CSV 網址的真實下載因測試環境 DNS 限制尚未通過；模擬下載測試已通過，不能視為正式網路驗證。
- 詳細測試範圍與限制見 `TEST_RESULTS.md`。尚未取得正式網址，不提供虛構的部署網址。

## 功能與新要求對照

| 要求 | 在哪裡看到 |
| --- | --- |
| Kaggle 公開資料，原始特徵數 > 13 | 題目① 洪水預測：20 個數值特徵，Kaggle `playground-series-s4e5` |
| CRISP-DM 六階段 | 網站「CRISP-DM」頁籤、原本 Colab Notebook |
| 多元線性回歸模型 | `ml_core.py` 使用 `sklearn.linear_model.LinearRegression` |
| 特徵選擇 | `SelectKBest(f_regression)`，可調 k 值 |
| 模型評估 + 預測圖/區間 | 「模型評估」頁籤：MAE、MSE、RMSE、R²、散佈圖、近似 95% 預測區間 |
| **加分 1：下拉選單選兩種 Problem** | 左側「預測題目」：① 洪水機率；② 50_Startups 獲利 |
| **加分 2：任意輸入 Dataset** | 左側選「公開 CSV 網址」，貼上公開 HTTPS CSV，並選擇 Target |
| Optional 報告/對話/資訊圖表 | `analysis/` 保留已完成的 Notebook、報告；私人 GPT 對話紀錄**不放公開 GitHub** |

> **特徵數規定說明**：主要選定的 Kaggle 洪水資料集有 20 個原始輸入特徵，符合「>13」要求。第二個加分題目採用老師題目示例所指的 50_Startups CSV，只有 4 個原始輸入特徵，僅作為第二個 problem 與自訂資料功能的示範；不宣稱它符合主題的 Kaggle 特徵數條件。

## 兩份內建資料

1. **洪水機率預測**：來自 Kaggle [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5/data)。為了讓公開網站快速回應，內建 `data/flood_sample.csv` 使用固定 random_state=42 從原本 1,117,957 筆訓練資料中抽取 **16,000 筆**，保留原始全部 20 個特徵。完整資料分析與最初 R² 結果請見 `analysis/7115056050_hw2.ipynb`，網站抽樣重訓結果可能略不同。
2. **新創公司獲利預測**：使用老師示例的 [50_Startups.csv](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv)，以 `Profit` 為 Target，三個數值輸入（R&D Spend、Administration、Marketing Spend）和 `State` 類別輸入。

## 任意資料集操作方式

1. 在左側選「公開 CSV 網址」。
2. 貼入公開的 HTTPS CSV URL（例如下方連結），或改選「上傳本機 CSV」。
3. 在主畫面「預測目標 Target」選擇要預測的連續數值欄位。
4. 調整 `SelectKBest` 要保留的特徵數量 `k`。
5. 查看資料統計、模型比較、散佈圖與預測區間；在「即時預測」輸入自己的數值。

老師示例 URL：

`https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`

網站為避免連到內部系統，只接受 **公開 HTTPS CSV**，檔案最大 **15 MB**；不支援私人 IP、localhost 或內網 URL。任意資料集應有至少 25 筆有效資料、連續數值 Target、至少一個有效特徵；高基數文字欄位會自動排除。對大型資料最多抽樣 50,000 筆供互動分析。

## GitHub + 網頁部署（主要繳交）

### 1. 建立 GitHub Repository

1. 進入 [GitHub](https://github.com/)，登入後按 `+` → `New repository`。
2. Repository name 建議：`7115056050-hw2`，可設為 `Public`。
3. 建立後選 `Add file` → `Upload files`。
4. **將解壓縮後專案資料夾裡面的檔案與子資料夾上傳到 repository 根目錄**，不要只上傳 ZIP。必須確保 `app.py` 和 `requirements.txt` 在根目錄，`data/flood_sample.csv` 與 `data/50_Startups.csv` 在 `data/`。
5. 點 `Commit changes`。

### 2. 部署網頁

1. 進入 [Streamlit Community Cloud](https://share.streamlit.io/) 並登入，連結 GitHub 帳號。
2. 選擇 `Create app` → `Yup, I have an app`。
3. 選 Repository `7115056050-hw2`、分支 `main`、主要程式檔 `app.py`。
4. 按 `Deploy`。平台會根據 `requirements.txt` 安裝 Python 套件。
5. 開啟平台給你的 `https://...streamlit.app/`，實測兩個問題、公開 CSV URL 及預測按鈕。

> Streamlit 官方部署文件：[Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

### 3. 作業繳交

- **GitHub URL**：`https://github.com/你的帳號/7115056050-hw2`
- **Web URL**：`https://你的網頁名稱.streamlit.app/`

以上是格式示意，實際網址要在你完成部署後填寫，**不要把示意網址直接交作業**。

## 在自己的電腦執行（選用）

安裝 Python 3.12 後，在專案資料夾開啟終端機：

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

如果是 Windows CMD，也可以使用：

```bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## 專案結構

```text
app.py
ml_core.py
requirements.txt
README.md
.gitignore
data/
  flood_sample.csv
  50_Startups.csv
analysis/
  7115056050_hw2.ipynb
  7115056050_hw2_報告.pdf
```

## 研究限制

- 洪水題目使用 Kaggle 競賽資料，網站輸出不是經過真實災害風險驗證的機率，不得作為正式洪水預警。
- 散佈圖與近似 95% 預測區間依賴回歸模型假設；實際涵蓋率不保證固定 95%。
- 任何模型的性能需以保留測試集評估，不能單從預測某一筆資料斷定準確度。
- 本次的 Ridge、Lasso 為原報告的文獻比較，**網頁實際訓練模型為 LinearRegression**；不要將 Ridge 或 Lasso 寫成已經實作並驗證的結果。

## 資料與參考來源

- Kaggle: https://www.kaggle.com/competitions/playground-series-s4e5/data
- Avik-Jain 50_Startups: https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv
- scikit-learn: https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn SelectKBest: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- Streamlit: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 本機驗證

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`test_app.py` 使用 Streamlit AppTest 測試兩種題目、特徵選擇、評估表、三張圖與自訂輸入預測。自訂 CSV URL 的自動測試明確模擬下載回應；正式 URL 可連線與公開網站驗收需另行實測。
