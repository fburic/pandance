# Unit tests for Pandance DataFrame operations
import datetime
from decimal import getcontext, Decimal
import math
import numpy as np
import pandas as pd

from hypothesis import assume, given, seed, strategies as st
import hypothesis.extra.numpy as hnp

import pandance as dance


def test_fuzzy_join_simple():
    df_a = pd.DataFrame([
        ('event1', 0.2),
        ('event2', 0.5),
        ('event3', 0.7),
        ('event4', 0.9)
    ], columns=['event', 'time_obs'])

    df_b = pd.DataFrame([
        ('event5', 0.1),
        ('event8', 0.89),
        ('event7', 0.8),
        ('event6', 0.54)
        ],
        columns=['event', 'time_obs'],
        index=list('abcd')
    )

    expected_result = pd.DataFrame([
        ('event2', 0.5, 'event6', 0.54),
        ('event4', 0.9, 'event8', 0.89)
    ], columns=['event_x', 'time_obs_x', 'event_y', 'time_obs_y'])

    theta_result = dance.theta_join(df_a, df_b, on='time_obs',
                                    relation=lambda x, y: abs(x-y) <= 0.05)
    theta_result_none = dance.theta_join(df_a, df_b, on='time_obs',
                                         relation=lambda x, y: abs(x-y) <= 0.01)

    fuzzy_result = dance.fuzzy_join(df_a, df_b, on='time_obs', tol=0.05)
    fuzzy_result_none = dance.fuzzy_join(df_a, df_b, on='time_obs', tol=0.001)

    assert fuzzy_result.compare(expected_result).empty
    assert fuzzy_result.compare(theta_result).empty
    assert fuzzy_result_none.shape[0] == 0
    assert theta_result_none.shape[0] == 0


def test_fuzzy_join_datetime():
    time_data_a = pd.DataFrame({
        'event': ['event1', 'event2', 'event3'],
        'obs_time': pd.to_datetime(
            ['2021-01-01 10:23', '2021-02-01 13:23', '2021-03-01 15:23']
        )
    })
    time_data_b = pd.DataFrame({
        'event': ['event4', 'event5', 'event6', 'event7'],
        'obs_time': pd.to_datetime(
            ['2021-01-01 10:22', '2021-02-01 21:23',  '2021-03-01 15:22', '2021-03-01 15:24']
        )
    })
    expected_result = pd.DataFrame([
        ('event1', '2021-01-01 10:23:00', 'event4', '2021-01-01 10:22:00'),
        ('event3', '2021-03-01 15:23:00', 'event6', '2021-03-01 15:22:00'),
        ('event3', '2021-03-01 15:23:00', 'event7', '2021-03-01 15:24:00')
    ], columns=['event_x', 'obs_time_x', 'event_y', 'obs_time_y'])
    expected_result['obs_time_x'] = pd.to_datetime(expected_result['obs_time_x'])
    expected_result['obs_time_y'] = pd.to_datetime(expected_result['obs_time_y'])

    fuzzy_result = (
        dance.fuzzy_join(time_data_a, time_data_b, on='obs_time',
                         tol=pd.Timedelta('1 minute'))
        .sort_values(['event_x', 'event_y'])
        .reset_index(drop=True)
    )
    theta_result = (
        dance.theta_join(time_data_a, time_data_b, on='obs_time',
                         relation=lambda ta, tb: abs(ta-tb) <= pd.Timedelta('1 minute'))
        .sort_values(['event_x', 'event_y'])
        .reset_index(drop=True)
    )
    assert fuzzy_result.compare(expected_result).empty
    assert fuzzy_result.compare(theta_result).empty


@given(
    data_range_start=st.datetimes(min_value=datetime.datetime(2022, 1, 1, 0, 0),
                                  max_value=datetime.datetime(2022, 1, 2, 0, 0),
                                  allow_imaginary=False),
    data_range_end=st.datetimes(min_value=datetime.datetime(2022, 1, 1, 0, 0),
                                max_value=datetime.datetime(2022, 1, 2, 0, 0),
                                allow_imaginary=False)
)
@seed(42)
def test_fuzzy_join_range_overlap(data_range_start, data_range_end):
    assume(data_range_start <= data_range_end)

    time_data_a = pd.DataFrame({
        'time': pd.date_range(data_range_start, data_range_end, freq='h')
    }).reset_index()

    time_data_b = pd.DataFrame({
        'time': pd.date_range(data_range_end - datetime.timedelta(),
                              data_range_end, freq='h')
    }).reset_index()

    fuzzy_result = (
        dance.fuzzy_join(time_data_a, time_data_b, on='time',
                         tol=pd.Timedelta('1 hour')).reset_index(drop=True)
        .sort_values(['index_x', 'index_y'])
        .reset_index(drop=True)
    )
    theta_result = (
        dance.theta_join(time_data_a, time_data_b, on='time',
                         relation=lambda ta, tb: abs(ta - tb) <= pd.Timedelta('1 hour'))
        .sort_values(['index_x', 'index_y'])
        .reset_index(drop=True)
    )
    assert fuzzy_result.compare(theta_result).empty


@given(
    values_a=hnp.arrays(np.float32, shape=(st.integers(1, 10))),
    tolerance=st.floats(min_value=np.finfo(np.float32).resolution, max_value=1),
)
@seed(42)
def test_fuzzy_join_safe(values_a, tolerance):
    """
    Safer test for mathematical correctness:
    Convert all values to Decimal to avoid float representation limitations
    """
    getcontext().prec = 128  # Very high precision
    df_a = pd.DataFrame(values_a, columns=['val']).reset_index()
    df_a['val'] = df_a['val'].map(lambda x: Decimal(x))

    # B values are obtained from A with deviations within tolerance
    df_b = pd.DataFrame(values_a, columns=['val']).reset_index()
    df_b['val'] = df_b['val'].map(lambda x: Decimal(x))
    tolerance = Decimal(tolerance)
    epsilon = Decimal(np.finfo(np.float32).eps.item())
    df_b['val'] = df_b['val'] + tolerance - epsilon

    fuzzy_result = dance.fuzzy_join(df_a, df_b, on='val', tol=tolerance)

    if np.isfinite(values_a).sum() == 0:
        assert fuzzy_result.shape[0] == 0

    else:
        result_row_correct = fuzzy_result.apply(
            lambda row: abs(row['val_x'] - row['val_y']) <= tolerance,
            axis='columns'
        )
        assert result_row_correct.all()

        theta_result = dance.theta_join(df_a, df_b, on='val',
                                        relation=lambda x, y: abs(x - y) <= tolerance)
        # Make DFs comparable
        fuzzy_result = (fuzzy_result.sort_values(['index_x', 'index_y'])
                        .reset_index(drop=True))
        theta_result = (theta_result.sort_values(['index_x', 'index_y'])
                        .reset_index(drop=True))
        assert fuzzy_result.compare(theta_result).empty


def test_fuzzy_join_type_combos():
    df_numeric = pd.DataFrame([1, 2, 3], columns=['val'])
    df_time = pd.DataFrame({
        'val': pd.date_range(datetime.datetime(2022, 1, 1, 0, 0),
                             datetime.datetime(2022, 1, 1, 5, 0), freq='h')
    })
    import pytest
    with pytest.raises(TypeError):
        dance.fuzzy_join(df_numeric, df_time, tol=0.1, on='val')
    with pytest.raises(TypeError):
        dance.fuzzy_join(df_time, df_time, tol=0.1, on='val')
    with pytest.raises(TypeError):
        dance.fuzzy_join(df_numeric, df_numeric, tol=pd.Timedelta('1 minute'), on='val')


def test_theta_join_numeric():
    a = pd.DataFrame.from_records(
        [
            (12, 10.1),
            (32, 20.2),
            (35, 30.3),
            (48, 40.4),
            (64, 50.5),
            (73, 60.6),
        ],
        columns=['key', 'value_old']
    )

    b = pd.DataFrame.from_records(
        [
            (18, 100.1),
            (30, 200.2),
            (32, 300.3),
            (64, 400.4),
            (78, 500.5),
            (96, 600.6),
        ],
        columns=['key', 'value_new']
    )

    result = dance.theta_join(a, b, on='key',
                              relation=lambda x, y: x % 32 == y % 32 == 0)

    a = a.assign(key_transf = a['key'] % 32)
    b = b.assign(key_transf = b['key'] % 32)
    expected_result = pd.merge(a, b, on='key_transf').drop(columns='key_transf')

    result = result.sort_values('key_x').reset_index(drop=True)
    expected_result = expected_result.sort_values('key_x').reset_index(drop=True)
    assert result.compare(expected_result).empty


def test_theta_join_relation():
    car = pd.DataFrame.from_records(
        [
            ('car_a', 20),
            ('car_b', 30),
            ('car_c', 50)
        ],
        columns=['item', 'price']
    )
    boat = pd.DataFrame.from_records(
       [
           ('boat_1', 10),
           ('boat_2', 40),
           ('boat_3', 60)
       ],
        columns=['item', 'price']
    )
    expected_result = pd.DataFrame.from_records(
        [
            ('car_a', 20, 'boat_1', 10),
            ('car_b', 30, 'boat_1', 10),
            ('car_c', 50, 'boat_1', 10),
            ('car_c', 50, 'boat_2', 40)
        ],
        columns=['item_old', 'price_old', 'item_new', 'price_new']
    )
    result = dance.theta_join(car, boat, on='price',
                              relation=lambda x, y: x >= y,
                              suffixes=('_old', '_new'))
    assert result.compare(expected_result).empty


def test_theta_join_strings():
    keywords = pd.DataFrame(['a', 'the', 'xyzzy'], columns=['keyword'])
    phrases = pd.DataFrame([
        'the quick brown fox jumps over the lazy dog',
        'lorem ipsum dolor'
    ], columns=['phrase'])
    expected_hits = pd.DataFrame([
        ('a', 'the quick brown fox jumps over the lazy dog'),
        ('the', 'the quick brown fox jumps over the lazy dog')
    ], columns=['keyword', 'phrase'])
    hits = dance.theta_join(
        keywords, phrases, left_on='keyword', right_on='phrase',
        relation=lambda kw, phrase: kw in phrase
    )
    assert hits.compare(expected_hits).empty


@given(
    angle=hnp.arrays(np.float32, shape=5,
                     elements=st.floats(width=32, allow_subnormal=True,
                                        allow_nan=True, allow_infinity=False))
)
@seed(42)
def test_theta_join_circle(angle):
    x = pd.DataFrame(np.cos(angle), columns=['x'])
    y = pd.DataFrame(np.sin(angle), columns=['y'])

    result = dance.theta_join(x, y, left_on='x', right_on='y',
                              relation=lambda x, y: math.isclose(x**2 + y**2 - 1, 0))
    vals = result.values
    assert np.allclose(np.power(vals[:, 0], 2) + np.power(vals[:, 1], 2) - 1, 0)


def test_mem_usage():
    len_a = 100
    len_b = 10
    unit_size_data = 2
    unit_size_idx = 8

    a = pd.DataFrame.from_records(
        np.arange(len_a, dtype=np.uint16).reshape(-1, 1),
        columns=['data']
    )

    b = pd.DataFrame.from_records(
        np.arange(100, 100 + len_b, dtype=np.uint16).reshape(-1, 1),
        columns=['data']
    )

    # Formulas derived for this test. Should be equivalent to generic func calc
    exp_idx_size = unit_size_idx * len_a * len_b
    exp_col_size = 2 * unit_size_data * len_a * len_b
    expected_size = (exp_idx_size + exp_col_size) / 1024**2
    if pd.__version__ < '1.4.0':
        expected_size *= 2

    cartesian_join = pd.merge(a[['data']], b[['data']], how='cross')
    cartesian_cost = cartesian_join.memory_usage(deep=True).sum() / 1024**2

    est_cost = dance._estimate_mem_cost_cartesian(a, b)
    # Triangle approximate equality just to be paranoid about float errors
    assert np.isclose(cartesian_cost, expected_size)
    assert np.isclose(expected_size, est_cost)
    assert np.isclose(cartesian_cost, est_cost)
