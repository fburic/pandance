import cProfile, pstats, io
import datetime
from pstats import SortKey

import numpy as np
import pandas as pd

import pandance as dance


def main():
    for func in [
        fuzzy_speed_random
    ]:
        profile_function(func)


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


if __name__ == '__main__':
    main()
