# Unit tests for Pandance DataFrame operations
import numpy as np
import pandas as pd

import pandance as dance


def test_fuzzy_join():
    pass


def test_theta_join_categorical():
    old_inventory = pd.DataFrame.from_records(
        [('drink', 12),
         ('sandwich', 40),
         ('starters', 25)],
        columns=['item', 'price']
    )

    new_inventory = pd.DataFrame.from_records(
        [('beverage', 14),
         ('sandwich', 45),
         ('appetizers', 25)],
        columns=['item', 'price']
    )

    synonymous = pd.DataFrame.from_records(
        [('drink', 'beverage'),
         ('sandwich', 'sandwich'),
         ('starters', 'appetizers')],
        columns=['old_name', 'new_name']
    )

    result = dance.theta_join(old_inventory, new_inventory, on='item',
                              relation=synonymous)

    expected_result = pd.DataFrame.from_records(
        [('drink', 12, 'beverage', 14),
         ('sandwich', 40, 'sandwich', 45),
         ('starters', 25, 'appetizers', 25)
         ],
        columns=['item_x', 'price_x', 'item_y', 'price_y']
    )
    assert result.compare(expected_result).empty


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

    cartesian_join = pd.merge(a[['data']], b[['data']], how='cross')
    cartesian_cost = cartesian_join.memory_usage(deep=True).sum() / 1024**2

    est_cost = dance._estimate_mem_cost_cartesian(a, b)
    # Triangle approximate equality just to be paranoid about float errors
    assert np.isclose(cartesian_cost, expected_size)
    assert np.isclose(expected_size, est_cost)
    assert np.isclose(cartesian_cost, est_cost)
