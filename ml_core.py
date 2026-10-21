"""HW2 regression engine. All learned transformations are fit on training data only."""
from dataclasses import dataclass
from io import BytesIO
from ipaddress import ip_address
from urllib.parse import urlparse, urljoin
import socket
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MAX_DOWNLOAD = 15 * 1024 * 1024
MAX_ROWS = 50_000


def _public_https(url: str):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('請提供公開 HTTPS CSV 網址（例如 raw.githubusercontent.com）。')
    try:
        results = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError('無法解析資料集主機名稱。') from exc
    for res in results:
        address = ip_address(res[4][0])
        if not address.is_global:
            raise ValueError('不允許存取內部網路、localhost 或私人 IP 位址。')


def download_public_csv(url: str) -> bytes:
    """Fetch a public HTTPS CSV with a size cap and redirect validation."""
    current = url.strip()
    session = requests.Session()
    try:
        for _ in range(4):
            _public_https(current)
            with session.get(current, timeout=(8, 20), allow_redirects=False, stream=True,
                             headers={'User-Agent': 'HW2-regression-demo/1.0'}) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect = response.headers.get('Location')
                    if not redirect:
                        raise ValueError('網址轉址缺少位置。')
                    current = urljoin(current, redirect)
                    continue
                response.raise_for_status()
                parts = []
                size = 0
                for chunk in response.iter_content(chunk_size=128 * 1024):
                    if chunk:
                        size += len(chunk)
                        if size > MAX_DOWNLOAD:
                            raise ValueError('CSV 檔案超過 15 MB，請改用較小的公開資料或抽樣 CSV。')
                        parts.append(chunk)
                return b''.join(parts)
        raise ValueError('網址轉址次數過多。')
    except requests.RequestException as exc:
        raise ValueError(f'資料下載失敗：{exc}') from exc
    finally:
        session.close()


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_DOWNLOAD:
        raise ValueError('CSV 檔案超過 15 MB 限制。')
    for enc in ('utf-8-sig', 'cp950', 'latin-1'):
        try:
            data_frame = pd.read_csv(BytesIO(data), encoding=enc, low_memory=False)
            if data_frame.shape[1] < 2:
                raise ValueError('CSV 至少需要一個特徵欄位與一個目標欄位。')
            return data_frame
        except UnicodeDecodeError:
            continue
    raise ValueError('無法讀取 CSV 檔案的文字編碼。')


@dataclass
class Results:
    source_rows: int
    sample_rows: int
    input_columns: list
    dropped_columns: list
    numeric_columns: list
    category_columns: list
    encoded_names: list
    feature_ranking: pd.DataFrame
    metrics: pd.DataFrame
    selected_names: list
    target: str
    X_test_original: pd.DataFrame
    y_test: np.ndarray
    pred_all: np.ndarray
    pred_selected: np.ndarray
    interval_lower: np.ndarray
    interval_upper: np.ndarray
    interval_coverage: float
    model_all: LinearRegression
    model_selected: LinearRegression
    encoder: ColumnTransformer
    selector: SelectKBest

    def predict_one(self, new_row: pd.DataFrame):
        row_enc = self.encoder.transform(new_row[self.input_columns])
        a = float(self.model_all.predict(row_enc)[0])
        b = float(self.model_selected.predict(self.selector.transform(row_enc))[0])
        return a, b


def analyze_regression(df: pd.DataFrame, target: str, k: int = 15, seed: int = 42) -> Results:
    if target not in df.columns:
        raise ValueError('選取的目標欄位不存在。')
    source_rows = len(df)
    work = df.copy().replace([np.inf, -np.inf], np.nan)
    work[target] = pd.to_numeric(work[target], errors='coerce')
    work = work.loc[work[target].notna()].copy()
    if len(work) < 25:
        raise ValueError('有效資料至少要有 25 筆，才能分割訓練集與測試集。')
    if work[target].nunique() < 3:
        raise ValueError('目前目標欄位不適合連續值迴歸（至少需要 3 個不同的數值）。')
    if len(work) > MAX_ROWS:
        work = work.sample(n=MAX_ROWS, random_state=seed)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        work.drop(columns=[target]), work[target], test_size=0.2, random_state=seed)
    initial = X_train_raw.columns.tolist()
    id_names = {'id', 'index', 'rowid', 'recordid', 'uuid', 'sample_id'}
    dropped = []
    numerical, categorical = [], []
    for c in initial:
        col = X_train_raw[c]
        if c.strip().lower().replace(' ', '') in id_names:
            dropped.append(c)
        elif col.nunique(dropna=True) < 2:
            dropped.append(c)
        elif pd.api.types.is_numeric_dtype(col):
            if col.nunique(dropna=True) < 2:
                dropped.append(c)
            else:
                numerical.append(c)
        elif col.nunique(dropna=True) <= 24:
            categorical.append(c)
        else:
            dropped.append(c)  # prevent one-hot explosion from free-text/high-cardinality ID columns
    inputs = numerical + categorical
    if not inputs:
        raise ValueError('找不到可用的數值或低基數類別特徵。')
    X_train, X_test = X_train_raw[inputs], X_test_raw[inputs]
    if len(numerical) + sum(X_train[c].nunique(dropna=True) for c in categorical) > 250:
        raise ValueError('類別展開後特徵過多，請改用較低基數的欄位。')

    pipes = []
    if numerical:
        pipes.append(('num', Pipeline([('fill', SimpleImputer(strategy='median'))]), numerical))
    if categorical:
        pipes.append(('cat', Pipeline([
            ('fill', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical))
    encoder = ColumnTransformer(pipes, remainder='drop', sparse_threshold=0)
    Xt = encoder.fit_transform(X_train)
    Xv = encoder.transform(X_test)
    names = [n.replace('num__', '').replace('cat__', '') for n in encoder.get_feature_names_out()]
    if Xt.shape[1] > 250:
        raise ValueError('類別展開後特徵過多，請改用較低基數的欄位。')
    k = min(max(int(k), 1), Xt.shape[1])
    selected = SelectKBest(score_func=f_regression, k=k)
    Xt_sel = selected.fit_transform(Xt, y_train)
    Xv_sel = selected.transform(Xv)
    full = LinearRegression().fit(Xt, y_train)
    reduced = LinearRegression().fit(Xt_sel, y_train)
    pred_full = full.predict(Xv)
    pred_reduced = reduced.predict(Xv_sel)
    actual = np.asarray(y_test)

    def met(pred):
        mse = mean_squared_error(actual, pred)
        return {'MAE': mean_absolute_error(actual, pred), 'MSE': mse,
                'RMSE': np.sqrt(mse), 'R²': r2_score(actual, pred)}
    metrics = pd.DataFrame({'全部特徵': met(pred_full), f'選擇 {k} 個特徵': met(pred_reduced)})

    # Approximate 95% observation prediction interval under classic OLS assumptions.
    # Use the training residual variance and leverage of each test point.
    train_residual = np.asarray(y_train) - full.predict(Xt)
    design = np.column_stack([np.ones(len(Xt)), Xt])
    rank = np.linalg.matrix_rank(design)
    variance = float(np.dot(train_residual, train_residual) / max(len(Xt) - rank, 1))
    cov = np.linalg.pinv(design.T @ design)
    test_design = np.column_stack([np.ones(len(Xv)), Xv])
    leverage = np.einsum('ij,jk,ik->i', test_design, cov, test_design)
    radius = 1.96 * np.sqrt(np.maximum(variance * (1 + leverage), 0))
    lo = pred_full - radius
    hi = pred_full + radius
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))

    scores = pd.DataFrame({'特徵': names, 'F 分數': selected.scores_})
    scores['F 分數'] = scores['F 分數'].replace([np.inf, -np.inf], np.nan)
    scores = scores.sort_values('F 分數', ascending=False, na_position='last')
    chosen = list(np.array(names)[selected.get_support()])
    return Results(source_rows, len(work), inputs, dropped, numerical, categorical,
                   names, scores, metrics, chosen, target, X_test, actual,
                   pred_full, pred_reduced, lo, hi, coverage,
                   full, reduced, encoder, selected)
