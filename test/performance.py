"""
Profile speed and memory usage of Pandance operations.
"""
import cProfile, pstats, io
import datetime
from pstats import SortKey
import random
import time

import numpy as np
import pandas as pd

import pandance as dance


def main():
    for func in [
        # fuzzy_speed_identical,
        # fuzzy_speed_random,
        # ineq_join_random_unif,
        # ineq_join_overlap_cartesian,
        theta_join_overlap_cartesian
    ]:
        profile_function(func)


def profile_function(func):
    prof = cProfile.Profile()

    prof.enable()
    func()
    prof.disable()

    stat_stream = io.StringIO()
    ps = (
        pstats.Stats(prof, stream=stat_stream)
        .sort_stats(SortKey.CUMULATIVE)
    )
    func_name = str(func).split(' ')[1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ps.dump_stats(func_name + '_' + timestamp + '.prof')
    print(stat_stream.getvalue())


def fuzzy_speed_identical():
    df_a = pd.DataFrame(
        np.ones(100),
        columns=['val']
    )
    df_b = pd.DataFrame(
        np.ones(1000),
        columns=['val']
    )
    _ = dance.fuzzy_join(df_a, df_b, on='val', tol=1)


def fuzzy_speed_random():
    """Overlap of two normal distributions"""
    rng = np.random.default_rng(12345)

    center_a = -2
    center_b = 2
    size_a = size_b = 10000

    df_a = pd.DataFrame(
        zip(range(size_a), rng.normal(center_a, 1, size_a)),
        columns=['idx', 'val']
    )

    df_b = pd.DataFrame(
        zip(range(size_b), rng.normal(center_b, 1, size_b)),
        columns=['idx', 'val']
    )
    _ = dance.fuzzy_join(df_a, df_b, on='val', tol=0.1)


def ineq_join_random_unif():
    """Overlap of two uniform distributions"""
    rng = np.random.default_rng(12345)

    size_a = size_b = 10000

    df_a = pd.DataFrame(
        zip(range(size_a), rng.uniform(0, 100, size_a)),
        columns=['idx', 'val']
    )

    df_b = pd.DataFrame(
        zip(range(size_b), rng.uniform(0, 100, size_b)),
        columns=['idx', 'val']
    )
    x = dance.ineq_join(df_a, df_b, on='val', how='<')
    print(x.shape[0])


def ineq_join_overlap_cartesian():
    """
    Generate two integer ranges A and B >= A with parametrized overlap length L
    The total number of pairs in the result is A * B  + Comb[L, 2] - L^2
    """
    len_a = 3000
    len_b = 3000
    len_overlap = 1500

    df_a = pd.DataFrame(range(0, len_a), columns=['val'])
    df_b = pd.DataFrame(range(len_a - len_overlap, len_a - len_overlap + len_b), columns=['val'])

    result = dance.ineq_join(df_a, df_b, on='val', how='<')


def theta_join_overlap_cartesian():
    """
    Generate two integer ranges A and B >= A with parametrized overlap length L
    The total number of pairs in the result is A * B  + Comb[L, 2] - L^2
    """
    len_a = 10
    len_b = 10
    len_overlap = 2

    df_a = pd.DataFrame(range(0, len_a), columns=['val'])
    df_b = pd.DataFrame(range(len_a - len_overlap, len_a - len_overlap + len_b), columns=['val'])

    result = dance.theta_join(
        df_a, df_b, on='val', 
        condition=lambda a, b: a < b,
        # condition=_sim_heavy_task,
        par_threshold = int(1e2),
        n_processes = 1
    )


def _sim_heavy_task(x, y):
    time.sleep(0.1)
    return random.choice([True, False])


if __name__ == '__main__':
    main()
