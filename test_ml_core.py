"""Quick offline smoke checks for both default problems and basic custom CSV."""
from pathlib import Path
import pandas as pd
from ml_core import analyze_regression

HERE = Path(__file__).parent


def test_flood():
    df = pd.read_csv(HERE / 'data' / 'flood_sample.csv')
    r = analyze_regression(df, 'FloodProbability', 15)
    assert len(r.input_columns) == 20
    assert len(r.selected_names) == 15
    assert r.metrics.loc['R²', '全部特徵'] > 0.7
    assert len(r.pred_all) == len(r.y_test)


def test_startups():
    df = pd.read_csv(HERE / 'data' / '50_Startups.csv')
    r = analyze_regression(df, 'Profit', 3)
    assert len(r.input_columns) == 4
    assert len(r.selected_names) == 3
    assert 'State' in r.category_columns
    assert r.predict_one(df[r.input_columns].iloc[[0]])[0] > 0


if __name__ == '__main__':
    test_flood()
    test_startups()
    print('PASS: flood / startup datasets, feature selection, evaluation, prediction')


def test_custom_csv_with_missing_and_infinite_values():
    import numpy as np
    df = pd.DataFrame({'x': np.arange(60, dtype=float),
                       'empty': pd.Series([None] * 60, dtype=object),
                       'y': np.arange(60, dtype=float) * 2 + 1})
    df.loc[0, 'x'] = np.inf
    df.loc[2, 'y'] = np.inf
    r = analyze_regression(df, 'y', 1)
    assert r.sample_rows == 59
    assert 'empty' in r.dropped_columns
    assert np.isfinite(r.metrics.to_numpy()).all()
    assert np.isfinite(r.predict_one(pd.DataFrame({'x': [10.0]}))).all()


def test_reject_invalid_csv_and_target():
    import pytest
    from ml_core import read_csv_bytes, _public_https
    with pytest.raises(ValueError):
        read_csv_bytes(b'only_column\n1\n2\n')
    with pytest.raises(ValueError):
        _public_https('http://localhost/data.csv')
    with pytest.raises(ValueError):
        analyze_regression(pd.DataFrame({'x': range(10), 'y': range(10)}), 'y')
