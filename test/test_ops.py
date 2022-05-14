# Unit tests for Pandance DataFrame operations
import pandas as pd

import pandance as dance


def test_fuzzy_join():
    pass


def test_theta_join_categori():

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
