"""Local UI tests. CSV network responses use an explicit fixture; not a live URL check."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).parent


def assert_rendered(at):
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.get('image')) == 3  # ranking, scatter comparison, prediction interval
    assert len(at.dataframe) >= 3
    assert np.isfinite(at.dataframe[2].value.to_numpy()).all()


def test_two_problems_features_and_predictions():
    at = AppTest.from_file(str(HERE / 'app.py'), default_timeout=60).run()
    assert_rendered(at)
    assert at.metric[0].value == '16,000'
    at.slider[0].set_value(5).run()
    assert_rendered(at)
    assert '選擇 5 個特徵' in at.dataframe[2].value.columns
    at.number_input[0].set_value(8.0)
    at.button[0].click().run()
    assert any(m.label == '模型 A 預測結果' for m in at.metric)
    at.sidebar.selectbox[0].set_value('② 新創公司獲利預測').run()
    assert_rendered(at)
    assert at.metric[0].value == '50'
    at.number_input[0].set_value(150000.0)
    at.button[0].click().run()
    assert all(np.isfinite(float(m.value.replace(',', ''))) for m in at.metric if '預測結果' in m.label)


def test_custom_url_and_single_feature_csv_fixture():
    data = pd.DataFrame({'x': range(50), 'y': np.arange(50) * 2 + 1}).to_csv(index=False).encode()
    at = AppTest.from_file(str(HERE / 'app.py'), default_timeout=60).run()
    with patch('ml_core.download_public_csv', return_value=data) as download:
        at.sidebar.radio[0].set_value('公開 CSV 網址').run()
        at.sidebar.text_input[0].set_value('https://example.com/custom-regression.csv').run()
        assert_rendered(at)
        download.assert_called_with('https://example.com/custom-regression.csv')
        assert not at.slider
        assert at.metric[0].value == '50'
        at.number_input[0].set_value(10.0)
        at.button[0].click().run()
        assert next(m.value for m in at.metric if m.label == '模型 A 預測結果') == '21.0000'
