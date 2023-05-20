import bisect
import datetime
import itertools
import logging
import operator
import warnings
from decimal import Decimal, InvalidOperation
from multiprocessing import Pool, cpu_count
from typing import Callable, Optional, Union, Iterable

import intervaltree as itree
import numpy as np
import pandas as pd
import psutil

__all__ = ['fuzzy_join', 'theta_join', 'ineq_join',
           '_estimate_mem_cost_cartesian', '_cartesian_join_with_mem_check']

logger = logging.getLogger()


def fuzzy_join(left: pd.DataFrame, right: pd.DataFrame,
               on: str = None, left_on: str = None, right_on: str = None,
               tol: Union[float, Decimal, pd.Timedelta] = 1e-3,
               suffixes: tuple = ('_x', '_y')) -> pd.DataFrame:
    """
    Perform an approximate inner join of two DataFrames, on a numerical or time column.
    For example, :math:`1.03 \\approx 1` would be a match, given an absolute tolerance ``tol = 0.5``.
    The tolerance is inclusive, meaning ``(a - b) <= tol`` is considered a match.

    A single join column may be used and this must be given explicitly
    (with ``on``, or ``left_on`` and ``right_on``).

    The joined DataFrame contains both columns that were used in the join,
    with appended ``suffixes``.

    .. warning::

        The matching may misbehave if numerical values are very large and the tolerance small,
        due to the simple absolute tolerance test and floating point representation
        limitations (see *Notes*).

    .. warning::

        `NaN` and `Inf` values in the joining column will (silently) not yield matches,
        as per the `IEEE 754 <https://en.wikipedia.org/wiki/NaN#Comparison_with_NaN>`_
        standard implemented by NumPy.

    .. note::

        This operation is a more efficient implementation
        compared to the generic :py:func:`theta_join`,
        taking an expected :math:`O((n + m) \\cdot \\log_2{m})` time,
        assuming the majority of values in the longer join column are distinct
        (accouting for the tolerance),
        where *m* is the length of the longer of the two DataFrames,
        and *n* of the shorter one, instead of :math:`O(n \\cdot m)`.

        The worst case of the fuzzy join is still :math:`O(n \\cdot m)` in
        the extreme case when both join colunms contain identical values
        (accounting for tolerance), meaning everything matches with everything.

    :param left: The left-hand side Pandas DataFrame
    :param right: The right-hand side Pandas DataFrame
    :param on: (Single) numerical or time column name to join on
    :param left_on: (Single) numerical or time column name to join on in the left DataFrame
    :param right_on: (Single) numerical or time column name to join on in the right DataFrame
    :param tol: Numerical or temporal tolerance for the fuzzy matching.
    :param suffixes: A length-2 sequence where each element is optionally
        a string indicating the suffix to add to overlapping column names
        in ``left`` and ``right``, respectively
    :return: The fuzzy join of the two DataFrames.

    Examples
    --------

    **Numerical columns**

    We have two sets of model performance scores (0..1).
    The models in one list are rather simple, the others much more complex.

    >>> import pandas as pd
    >>> simple_models = pd.DataFrame([
    ...     ('A', 0.2),
    ...     ('B', 0.5),
    ...     ('C', 0.7),
    ...     ('D', 0.9)
    ... ], columns=['model', 'score'])
    >>> complex_models = pd.DataFrame([
    ...     ('M1', 0.1),
    ...     ('M2', 0.89),
    ...     ('M3', 0.8),
    ...     ('M4', 0.54)
    ... ], columns=['model', 'score'])

    We're interested in finding models that perform essentially the same
    across the two lists, accepting a score tolerance of 0.05:

    >>> import pandance as dance
    >>> dance.fuzzy_join(simple_models, complex_models, on='score', tol=0.05, suffixes=('_s', '_f'))
      model_s  score_s model_f  score_f
    0       B      0.5      M4     0.54
    1       D      0.9      M2     0.89

    **Time series**

    Given two datasets recording the observation times of events,
    perform a fuzzy join on the time column
    to get only the events that occur at approximately the same time between sets::

      df_x:                                df_y:

        +--------+---------------------+     +--------+---------------------+
        |  event |            obs_time |     |  event |            obs_time |
        +--------+---------------------+     +--------+---------------------+
        | event1 | 2021-01-01 10:23:00 |     | event4 | 2021-01-01 10:22:00 |
        | event2 | 2021-02-01 13:23:00 |     | event5 | 2021-02-01 21:23:00 |
        | event3 | 2021-03-01 15:23:00 |     | event6 | 2021-03-01 15:22:00 |
        +--------+---------------------+     | event7 | 2021-03-01 15:24:00 |
                                             +--------+---------------------+

    The operation::

        fuzzy_join(df_x, df_y, on='obs_time', tol=pd.Timedelta('1 minute'))

    gives::

        +---------+---------------------+---------+---------------------+
        | event_x |          obs_time_x | event_y |          obs_time_y |
        +---------+---------------------+---------+---------------------+
        | event1  | 2021-01-01 10:23:00 | event4  | 2021-01-01 10:22:00 |
        | event3  | 2021-03-01 15:23:00 | event6  | 2021-03-01 15:22:00 |
        | event3  | 2021-03-01 15:23:00 | event7  | 2021-03-01 15:24:00 |
        +---------+---------------------+---------+---------------------+

    Notes
    -----

    **High-precision applications**

    Care must be taken if high precision (low tolerances) are to be used with
    large floating point numbers, due to representation limitations.

    As a workaround, consider using arbitrary precision data types, such as the
    Python built-in `Decimal <https://docs.python.org/3/library/decimal.html>`_ type,
    accepting the performance penalty.
    The join columns can be converted to ``Decimal`` just before the fuzzy join operation::

        from decimal import Decimal
        import pandance as dance

        df_a['val'] = df_a['val'].map(lambda x: Decimal(x))
        df_b['val'] = df_b['val'].map(lambda x: Decimal(x))

        dance.fuzzy_join(df_b, df_a, on='val', tol=Decimal(1e-10))

    See the ``decimal`` documentation on setting the precision (number of decimals)
    to be used for results of operations with Decimals.

    A more widespread practice is to adjust float comparisons with
    an additional relative tolerance,
    but there is no straightforward way to do this for this join operation,
    since we're (conceptually) symmetrically comparing everything with everything
    in the left and right join columns.
    `Comparisons with relative tolerance <https://docs.python.org/3/library/math.html#math.isclose>`_
    factor in both numbers to be compared.
    For more technical details on the issue, see this
    `post <https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/>`_.
    """
    left_on, right_on = _validate_input_col_names(on, left_on, right_on)
    left, right = _def_validate_and_clean_inputs_to_fuzzy(left, right,
                                                          left_on, right_on, tol)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return _empty_df(left_on, right_on, suffixes)

    if left.shape[0] <= right.shape[0]:
        swap_col_order = False
        shorter_col, longer_col = left_on, right_on
        shorter_df, longer_df = left, right
    else:
        swap_col_order = True
        longer_col, shorter_col = left_on, right_on
        longer_df, shorter_df = left, right

    if isinstance(tol, (np.timedelta64, pd.Timedelta, datetime.timedelta)):
        epsilon = pd.Timedelta('1 ns')
    else:
        epsilon = np.finfo(np.float32).eps
    if isinstance(left[left_on].values[0], Decimal):
        tol = Decimal(tol)
        epsilon = Decimal(epsilon.item())

    interval_tree = _build_interval_tree(longer_df[[longer_col]], tol, epsilon)
    index_association = _get_fuzzy_match_indices(shorter_df[[shorter_col]], interval_tree)
    index_association = list(index_association)
    if not index_association:
        return _empty_df(left_on, right_on, suffixes)
    index_assoc_short, index_assoc_long = zip(*index_association)

    # Merge on new index to match order of associated left-right indices
    rows_short = shorter_df.loc[(i for i in index_assoc_short)].reset_index(drop=True)
    rows_long = longer_df.loc[(i for i in index_assoc_long)].reset_index(drop=True)

    # Reflect order of input DataFrames
    if swap_col_order:
        rows_short, rows_long = rows_long, rows_short
    join_result = rows_short.join(rows_long, lsuffix=suffixes[0], rsuffix=suffixes[1])
    return join_result


def _build_interval_tree(col_df: pd.DataFrame,
                         interval_radius: float,
                         epsilon: float = np.finfo(np.float32).eps) -> itree.IntervalTree:
    """
    Build a self-balancing interval tree from the single-column DataFrame `col_df`,
    with intervals constructed as ``[x - interval_radius, x + interval_radius + epsilon)``,
    for all x in `col_df`.
    The `epsilon` is added to make the interval inclusive
    (IntervalTree doesn't have the option), in order to avoid asymmetries in the
    join operation (e.g. ``fuzzy_join(1, 2, tol=1)`` gives 1 falling in ``[1, 3)``
    but ``fuzzy_join(2, 1, tol=1)`` has 2 not falling in ``[0, 2)``).

    :param col_df: DataFrame with single numerical column
    :param interval_radius: How wide the intervals around values should be
    """
    interval_tree = itree.IntervalTree()
    colname = col_df.columns[0]
    col_df.apply(
        lambda row: interval_tree.addi(row[colname] - interval_radius,
                                       row[colname] + interval_radius + epsilon,
                                       data=row.name),
        axis='columns'
    )

    # Identical intervals are removed and their indices are merged into a list
    interval_tree.merge_equals(
        data_reducer=lambda idx_x, idx_y: [idx_y] if idx_x is None else idx_x + [idx_y],
        data_initializer=[]
    )
    return interval_tree


def _get_fuzzy_match_indices(df_col: pd.DataFrame,
                             interval_tree: itree.IntervalTree) -> Iterable:
    colname = df_col.columns[0]
    index_assoc = df_col.apply(
        lambda row: _matching_indices_for_value(row[colname], row.name, interval_tree),
        axis='columns'
    )
    return itertools.chain.from_iterable(index_assoc.values)


def _matching_indices_for_value(val, val_idx, interval_tree: itree.IntervalTree):
    # if _is_valid_value(val):
    for m_interval in interval_tree[val]:
        m_idx = m_interval.data
        for i in m_idx:
            yield val_idx, i


ToleranceType = Union[int, float, np.integer, np.floating,
                      np.timedelta64, pd.Timedelta]


def _def_validate_and_clean_inputs_to_fuzzy(left: pd.DataFrame,
                                            right: pd.DataFrame,
                                            left_on: str,
                                            right_on: str,
                                            tol: ToleranceType) -> tuple:
    supported_dtypes = ['i', 'u', 'f']
    exception_msg = f'Operation only supports joining on columns ' \
                    f'of NumPy types: {supported_dtypes}, ' \
                    f'NumPy datetime64, or decimal.Decimal'

    left_val_sample = left[left_on].values[0]
    right_val_sample = right[right_on].values[0]

    if left[left_on].dtype.kind not in supported_dtypes:
        if left.shape[0] > 0:
            if not isinstance(left_val_sample, (Decimal, np.datetime64)):
                raise TypeError('Left DataFrame invalid: ' + exception_msg)

    if right[right_on].dtype.kind not in supported_dtypes:
        if right.shape[0] > 0:
            if not isinstance(right_val_sample, (Decimal, np.datetime64)):
                raise TypeError('Right DataFrame invalid: ' + exception_msg)

    if isinstance(left_val_sample, np.datetime64) ^ isinstance(right_val_sample, np.datetime64):
        raise TypeError('Both columns must be datetime64')

    if (isinstance(left_val_sample, np.datetime64)
            and not isinstance(tol, (pd.Timedelta, datetime.timedelta))):
        raise TypeError('When working with datetime64, the tolerance must be '
                        'numpy.timedelta64 or datetime.timedelta')

    left = left[left[left_on].map(_is_valid_value)]
    right = right[right[right_on].map(_is_valid_value)]
    return left, right


def _is_valid_value(val: Union[np.floating, float, Decimal]) -> bool:
    """Non-finite values (NaNs and Inf) are not valid"""
    if (isinstance(val, np.floating)
            or isinstance(val, np.integer)
            or isinstance(val, np.unsignedinteger)):
        return np.isfinite(val)

    elif isinstance(val, Decimal):
        return val.is_finite()

    else:
        return True


def _empty_df(left_on, right_on, suffixes):
    if left_on == right_on:
        left_on, right_on = left_on + suffixes[0], right_on + suffixes[1]
    return pd.DataFrame([], columns=[left_on, right_on])


def theta_join(left: pd.DataFrame, right: pd.DataFrame,
               condition: Callable[..., bool] = None,
               on: str = None, left_on: str = None, right_on: str = None,
               suffixes: Optional[tuple] = ('_x', '_y'),
               n_processes: int = None,
               par_threshold: int = 1000,
               relation: Callable[..., bool] = None) -> pd.DataFrame:
    """
    Perform an inner join with a user-specified match ``condition``.

    A *theta-join* is an operation in which rows in the join columns
    are matched using an arbitrary condition :math:`\\theta`
    that holds between the row entries
    (i.e. the pair is in a `binary relation <https://en.wikipedia.org/wiki/Binary_relation>`_).
    It generalizes equijoins (where :math:`\\theta` = equality).
    See examples below and the
    `Wikipedia article <https://en.wikipedia.org/wiki/Relational_algebra#%CE%B8-join_and_equijoin>`_,
    though in Pandance, :math:`\\theta` is not limited to the typical set of relations
    {<, <=, =, !=, >=, >}. Rather, the user may specify any boolean-valued function
    as a ``condition``, as described below.

    Since version ``0.3.0``, this join is parallelized for larger results.
    To avoid unnecessary overhead on small data,
    multiple processes are used only if the number of rows
    in the intermediate Cartesian join (= left x right lengths)
    is at least ``par_threshold``.
    *Consider decreasing this threshold* if the condition function
    takes a longer time to evaluate
    (complex calculation, some sort of lookup / query, etc.)

    By default, all CPU cores on the machine are used.

    .. warning::
        This operation is **memory-intensive!**
        Since this is a generic operation for any given `theta` condition,
        it's implemented as a Cartesian product of the two ``on`` columns
        in the input DataFrames,
        followed by a filter on the pairs for which the `theta` condition holds.
        So the memory usage is :math:`O(n \\cdot m)`,
        where `n` and `m` are the respective sizes of the ``on`` columns.

        A warning is logged if the estimated requirement is above 75%
        of available memory and a ``MemoryError`` is raised if the estimate exceeds
        available memory.

    :param left: The left-hand side Pandas DataFrame
    :param right: The right-hand side Pandas DataFrame
    :param condition: a **function** or callable object
        of two parameters ``x``, ``y`` that returns ``True``
        if the pair ``(x, y)`` fulfills the condition, else ``False``.
        E.g. ``divides(2, 8) == True``.
    :param on: (Single) column name to join on, passed to ``pandas.merge()``
    :param left_on: (Single) column name to join on in the left DataFrame,
        passed to ``pandas.merge()``
    :param right_on: (Single) column name to join on in the right DataFrame,
        passed to ``pandas.merge()``
    :param suffixes: A length-2 sequence where each element is optionally
        a string indicating the suffix to add to overlapping column names
        in left and right respectively, passed to ``pandas.merge()``
    :param relation: (*deprecated*) Synonym for ``condition``
    :param par_threshold: The intermediate Cartesian (cross) join must have at least
                          this many rows for parallelism to be used when filtering
                          on the *theta* condition.

                          .. versionadded:: 0.3.0

    :param n_processes: How many processes to spawn for performing the join.
                        Defaults to the number of CPUs on the system.

                        .. versionadded:: 0.3.0

    :return: The *theta*-join of the two DataFrames.


    .. seealso::

        :py:func:`fuzzy_join`
            An efficient implementation of the special case of θ-join where θ is :math:`\\approx`.

        :py:func:`ineq_join`
            An efficient implementation of the special case of θ-join
            where θ is an inequality {<, <=, =, >=, >} between the join columns.


    Examples
    --------

    **Numerical condition**

    We have two tables with numerical entries x and y,
    and we want to find those combinations of x and y that
    represent coordinates on the unit circle. Here

    .. math:: \\theta (x, y): x^2 + y^2 - 1 = 0

    >>> import math
    >>> import pandas as pd
    >>> import pandance as dance
    ...
    >>> horiz_vals = pd.DataFrame([0, 1, -1, 0.5], columns=['x'])
    >>> vert_vals = pd.DataFrame([0, 1, -1, 0.5], columns=['y'])
    ...
    >>> dance.theta_join(
    ...     horiz_vals, vert_vals, left_on='x', right_on='y',
    ...     condition = lambda x, y: math.isclose(x**2 + y**2 - 1, 0)
    ... )
         x    y
    0  0.0  1.0
    1  0.0 -1.0
    2  1.0  0.0
    3 -1.0  0.0


    **Substring matching**

    We have two tables of character strings and want to find all pairs in which
    strings from the left join column appear as substrings of the right.

    >>> import pandas as pd
    >>> import pandance as dance
    ...
    >>> keywords = pd.DataFrame(['a', 'the', 'xyzzy'], columns=['keyword'])
    >>> phrases = pd.DataFrame([
    ...     'the quick brown fox jumps over the lazy dog',
    ...     'lorem ipsum dolor'
    ... ], columns=['phrase'])

    Here `θ(a, b): a substring of b`.

    >>> hits = dance.theta_join(
    ...     keywords, phrases, left_on='keyword', right_on='phrase',
    ...     condition = lambda kw, phrase: kw in phrase)
    >>> hits
      keyword                                       phrase
    0       a  the quick brown fox jumps over the lazy dog
    1     the  the quick brown fox jumps over the lazy dog


    **Inequality condition**

    We're making a groceries list, and we're balancing macronutrients and costs.

    >>> import pandas as pd
    >>> import pandance as dance
    ...
    >>> carb_sources = pd.DataFrame([
    ...     ('rice', 34),
    ...     ('oat flakes', 32)
    ... ], columns=['item', 'price'])
    ...
    >>> protein_sources = pd.DataFrame([
    ...     ('lentils', 25),
    ...     ('chickpeas', 38),
    ...     ('soy beans', 48)
    ... ], columns=['item', 'price'])

    We want to stock up on a single carb and protein source,
    but we *want the carbs to cost less than the proteins*.
    This can be expressed as the θ-join below, where

    .. math:: \\theta (x, y): x < y

    >>> dance.theta_join(
    ...     carb_sources, protein_sources, on='price',
    ...     condition = lambda price_carb, price_prot: price_carb < price_prot,
    ...     suffixes=('_carb', '_prot'))
    ...
        item_carb  price_carb  item_prot  price_prot
    0        rice          34  chickpeas          38
    1        rice          34  soy beans          48
    2  oat flakes          32  chickpeas          38
    3  oat flakes          32  soy beans          48

    .. tip::
        The :py:func:`ineq_join` operation implements a more efficient version
        of this type of inequality join.
    """
    if condition is None and relation is None:
        raise Exception('Missing condition.')
    if condition is None:
        warnings.warn("theta_join(): The 'relation' argument is deprecated. "
                      "Use 'condition' instead. "
                      "This will be an error in pandance 0.4.0",
                      DeprecationWarning,
                      stacklevel=2)
        condition = relation

    left_on, right_on = _validate_input_col_names(on, left_on, right_on)

    result = _cartesian_join_with_mem_check(left, right, left_on, right_on, suffixes)

    def _safe_condition(x, y) -> bool:
        """Wrapper to guard against known exceptions"""
        try:
            return condition(x, y)
        except InvalidOperation:
            return False

    # Filter on theta
    if left_on == right_on:
        left_on, right_on = left_on + suffixes[0], right_on + suffixes[1]

    if n_processes is None:
        n_processes = cpu_count()

    if n_processes > 1 and result.shape[0] >= par_threshold:
        result_pairs = result[[left_on, right_on]].to_records(index=False)
        with Pool(processes=n_processes,
                  initializer=_theta_filter_init,
                  initargs=(condition,)) as pool:
            row_matches = pool.starmap(
                _safe_condition_parallel,
                result_pairs
            )
    else:
        row_matches = result.apply(
            lambda row: _safe_condition(row[left_on], row[right_on]),
            axis='columns'
        )
    result = result[row_matches]
    if result.shape[0] == 0:
        return result

    # Get other column items from input DataFrames
    result = pd.merge(
        left.loc[result['index' + suffixes[0]]].reset_index(drop=True),
        right.loc[result['index' + suffixes[1]]].reset_index(drop=True),
        left_index=True, right_index=True, suffixes=suffixes
    )
    return result


def _safe_condition_parallel(x, y) -> bool:
    """
    Wrapper to guard against known exceptions,
    as a module-level function to work with multiprocessing.
    """
    try:
        return _theta_condition(x, y)
    except InvalidOperation:
        return False


_theta_condition = None


def _theta_filter_init(func: Callable):
    """
    A hack to wrap lambdas (user-provided conditions),
    so we can multiprocess filtering on the condition.
    Source: https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
    """
    global _theta_condition
    _theta_condition = func


def _cartesian_join_with_mem_check(left: pd.DataFrame, right: pd.DataFrame,
                                   left_on: str, right_on: str, suffixes: tuple) -> pd.DataFrame:
    """
    Wraps a Pandas cross (Cartesian) join with a memory usage check.
    Throw ``MemoryError`` if estimated usage exceeds available memory.
    """
    est_mem = _estimate_mem_cost_cartesian(left[[left_on]], right[[right_on]])
    avail_mem = psutil.virtual_memory()
    avail_mem = (avail_mem.total - avail_mem.used) / 1024**2
    if est_mem > avail_mem:
        logger.error(f'The operation requires more memory than is currently available: {est_mem} MiB')
        raise MemoryError
    if est_mem / avail_mem > 0.75:
        logger.warning(f'The operation requires over 75% ({est_mem}) of available memory')

    # Cartesian join
    result = pd.merge(left[[left_on]].reset_index(),
                      right[[right_on]].reset_index(),
                      how='cross',
                      suffixes=suffixes)
    return result


def ineq_join(left: pd.DataFrame, right: pd.DataFrame,
              how: str = '<=', on: str = None, left_on: str = None, right_on: str = None,
              suffixes: Optional[tuple] = ('_x', '_y')) -> pd.DataFrame:
    """
    Perform an inequality join on the (single) specified left and right columns,
    for example::

        ineq_join(df_a, df_b, '<=', on='value')

    will match all ``(df_a['value'] <= df_b['value'])`` pairs.

    Valid column types are those that support comparisons (numbers, strings, dateime, etc.).

    Note that the operation is not guaranteed to preserve the row or column order
    of the input dataframes, in order to save time.

    .. note::

        The time cost of this operation is
        :math:`O\\left(n \\log_2{m} + m\\log_2{m} + Q \\right)`,
        where *n* and *m* are the lengths of the shorter and longer dataframe, respectively,
        and *Q* is the total number of matching entries in the join.
        Thus, the *worst-case* time cost is :math:`O(n \\cdot m)`, which happens
        when all values on the right-hand side of e.g. ``<`` are larger than those on
        the left-hand side (resulting in everything matching with everything),
        and analogously for ``>``.
        When *Q* is small (or constant with respect to growing *n* and *m*),
        the time cost becomes :math:`O((n + m) \\cdot \\log_2{m})`.

    :param left: The left-hand side Pandas DataFrame
    :param right: The right-hand side Pandas DataFrame
    :param how: The inequality operator between the two columns.
                Can be ``<``, ``<=``, ``>=``, or ``>``.
    :param on: (Single) column name to join on, passed to ``pandas.merge()``
    :param left_on: (Single) column name to join on in the left DataFrame,
        passed to ``pandas.merge()``
    :param right_on: (Single) column name to join on in the right DataFrame,
        passed to ``pandas.merge()``
    :param suffixes: A length-2 sequence where each element is optionally
        a string indicating the suffix to add to overlapping column names
        in left and right respectively, passed to ``pandas.merge()``
    :return: The inequijoin of the two DataFrames.

    .. seealso::

        :py:func:`theta_join` : A gneric join that allows user-specified joining conditions.

    Examples
    --------

    **Temporal data**

    Say we want to find connecting flights
    between locations A and C with a layover in location B.
    We have a table with flights from A to B, and another from B to C:
    Valid connections are those flights that arrive in B before a departing flight from B:

    >>> import pandas as pd
    >>> import pandance as dance
    ...
    >>> flights_ab = pd.DataFrame([
    ...     ('2023-01-01 08:00', '2023-01-01 10:00'),
    ...     ('2023-01-01 12:00', '2023-01-01 14:00'),
    ...     ('2023-01-01 16:00', '2023-01-01 18:00'),
    ...     ('2023-01-01 20:00', '2023-01-01 22:00')
    ... ], columns=['dep', 'arr'], dtype='datetime64[ns]')
    ...
    >>> flights_bc = pd.DataFrame([
    ...     ('2023-01-01 09:00', '2023-01-01 12:00'),
    ...     ('2023-01-01 14:00', '2023-01-01 17:00'),
    ...     ('2023-01-01 18:00', '2023-01-01 21:00'),
    ...     ('2023-01-01 21:00', '2023-01-02 00:00')
    ... ], columns=['dep', 'arr'], dtype='datetime64[ns]')
    ...
    >>> dance.ineq_join(flights_ab, flights_bc,
    ...                 left_on='arr', right_on='dep', how='<',
    ...                 suffixes=('_ab', '_bc'))
    ...
                   dep_ab                 arr_ab                  dep_bc                  arr_bc
    0 2023-01-01 08:00:00    2023-01-01 10:00:00     2023-01-01 14:00:00     2023-01-01 17:00:00
    0 2023-01-01 08:00:00    2023-01-01 10:00:00     2023-01-01 18:00:00     2023-01-01 21:00:00
    1 2023-01-01 12:00:00    2023-01-01 14:00:00     2023-01-01 18:00:00     2023-01-01 21:00:00
    0 2023-01-01 08:00:00    2023-01-01 10:00:00     2023-01-01 21:00:00     2023-01-02 00:00:00
    1 2023-01-01 12:00:00    2023-01-01 14:00:00     2023-01-01 21:00:00     2023-01-02 00:00:00
    2 2023-01-01 16:00:00    2023-01-01 18:00:00     2023-01-01 21:00:00     2023-01-02 00:00:00

    **Numerical data**

    We're making a groceries list, and we're balancing macronutrients and costs.

    >>> carb_sources = pd.DataFrame([
    ...     ('rice', 34),
    ...     ('oat flakes', 32)
    ... ], columns=['item', 'price'])
    ...
    >>> protein_sources = pd.DataFrame([
    ...     ('lentils', 25),
    ...     ('chickpeas', 38),
    ...     ('soy beans', 48)
    ... ], columns=['item', 'price'])

    We want to stock up on a single carb and protein source,
    but we *want the carbs to cost less than the proteins*.
    This can be expressed as the following inequality join:

    >>> dance.ineq_join(
    ...     carb_sources, protein_sources, on='price', how = '<',
    ...     suffixes=('_carb', '_prot'))
    ...
        item_carb  price_carb  price_prot  item_prot
    1        rice          34          38  chickpeas
    1  oat flakes          32          38  chickpeas
    2        rice          34          48  soy beans
    2  oat flakes          32          48  soy beans


    **Strings**

    Suppose we have a small sample of strings and want to find all strings that are
    sorted lower in a large database of strings (which here is constructed randomly).

    >>> import random
    >>> import string
    >>> random.seed(42)
    ...
    >>> query = pd.DataFrame(['bbb', 'ccc'], columns=['string'])
    ...
    >>> database = pd.DataFrame(
    ...     [''.join(random.choices(string.ascii_lowercase, k=3)) for _ in range(10)],
    ...     columns=['string']
    ... )

    In this case, the random database only has a few strings of lower ordering than our query

    >>> dance.ineq_join(query, database, how='>', on='string', suffixes=('_query', '_db'))
      string_query string_db
    3          bbb       afn
    3          ccc       afn
    4          bbb       afq
    4          ccc       afq
    """
    allowed_rels = ['<', '<=', '>=', '>']
    opposite_rel = {'<': '>', '<=': '>=', '>': '<', '>=': '<='}
    cmp_op = {'<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge}

    if how not in allowed_rels:
        raise ValueError('The inequality "how" relation can only be: ',
                         ', '.join(allowed_rels))
    left_on, right_on = _validate_input_col_names(on, left_on, right_on)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return _empty_df(left_on, right_on, suffixes)

    if left.shape[0] <= right.shape[0]:
        shorter_col, longer_col = left_on, right_on
        shorter_df, longer_df = left, right
    else:
        longer_col, shorter_col = left_on, right_on
        longer_df, shorter_df = left, right
        suffixes = suffixes[::-1]
        how = opposite_rel[how]

    # We take advantage of the transitive nature of the operator on join values.
    # Taking A (shorter df, "query") < B (longer df, "lookup"):
    #
    # 1. Sort lookup
    # 2. For each entry a in A, search for first b in B s.t. a < b
    #    using binary search (worst case: log B)
    # 3. A match a < b implies all B entries y > b match => add (a, y:B) pairs
    #
    # We could in principle symmetrically add all pairs (x:A, y:B) where x <= a and y <=b,
    # but it turned out that the pure Python implementation of that procedure
    # was x2 slower than just mapping the binary search lookup to the A dataframe join column.
    query = shorter_df[[shorter_col]].reset_index()
    query = query.rename(columns={'index': 'orig_idx' + suffixes[0]})
    lookup = longer_df[[longer_col]].sort_values(longer_col).reset_index()
    lookup = lookup.rename(columns={'index': 'orig_idx' + suffixes[1]})

    query_min, query_max = query[shorter_col].min(), query[shorter_col].max()
    lookup_min, lookup_max = lookup[longer_col].iloc[0], lookup[longer_col].iloc[-1]
    ranges_overlap = query_min <= lookup_max and query_max >= lookup_min
    if not ranges_overlap:
        # If the relation between the left and right (query and lookup) intervals
        # is the same as the "how" operator (e.g. how = '<', query < lookup),
        # the result is a Cartesian join (everything matches),
        # otherwise it's the empty set (nothing matches).
        # No interval overlap guarantees:
        # - safe to allow for -or-equal operators as well
        # - no need to switch interval ends between checking (query < lookup) and (query > lookup)
        if cmp_op[how](query_max, lookup_min):
            return _cartesian_join_with_mem_check(left, right, left_on, right_on, suffixes)
        else:
            merge_cols = _get_join_column_names(left, right, suffixes)
            return pd.DataFrame([], columns=merge_cols)

    # Try to save memory (overestimate result size as it's unknown at this point, but we need the type)
    idx_type = 'int32' if query.shape[0] * lookup.shape[0] < np.iinfo(np.int32).max else 'int64'

    if how[0] == '<':
        if how == '<':
            match_entries = _match_higher_values_right
        else:
            match_entries = _match_higher_values_left
    else:
        if how == '>':
            match_entries = _natch_lower_values_left
        else:
            match_entries = _natch_lower_values_right

    matches_flattened = np.vstack(list(map(
        lambda row: _get_ineq_match_results(match_entries, lookup, longer_col, row, idx_type),
        query.values
    )))
    matches_flattened = pd.DataFrame(
        {'orig_idx' + suffixes[0]: matches_flattened[:, 0]},
        index=matches_flattened[:, 1]
    )

    result = matches_flattened.join(lookup, how='inner', lsuffix=suffixes[0], rsuffix=suffixes[1])

    result.set_index('orig_idx' + suffixes[0], inplace=True)
    result = shorter_df.join(
        result, how='inner',
        lsuffix=suffixes[0], rsuffix=suffixes[1]
    )

    result.set_index('orig_idx' + suffixes[1], inplace=True)
    result = result.join(
        longer_df.drop(columns=longer_col), how='inner',
        lsuffix=suffixes[0], rsuffix=suffixes[1]
    )

    return result


def _get_ineq_match_results(match_entries: Callable, lookup: pd.DataFrame, longer_col: str,
                            row, idx_type) -> np.ndarray:
    """
    Find all matching lookup entries for current query row,
    then unroll ("explode"/"flatten") list of matches to an array of index pairs.
    row = [orig_idx, shorter join col value]
    """
    matched_indices = match_entries(row[1], lookup, longer_col)
    matched_indices_arr = np.zeros((len(matched_indices), 2), dtype=idx_type)
    matched_indices_arr[:, 0] = row[0]
    matched_indices_arr[:, 1] = matched_indices
    return matched_indices_arr


def _match_higher_values_left(val, lookup, longer_col):
    return range(bisect.bisect_left(lookup[longer_col].values, val), lookup.shape[0])


def _match_higher_values_right(val, lookup, longer_col):
    return range(bisect.bisect_right(lookup[longer_col].values, val), lookup.shape[0])


def _natch_lower_values_left(val, lookup, longer_col):
    return range(0, bisect.bisect_left(lookup[longer_col].values, val))


def _natch_lower_values_right(val, lookup, longer_col):
    return range(0, bisect.bisect_right(lookup[longer_col].values, val))


def _estimate_mem_cost_cartesian(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """
    Return the estimated memory usage (in MiB) of the Cartesian join
    of the two single-column DataFrames ``a`` and ``b``.
    The calculation uses deep memory measurement and includes the indices.

    :param a: A single-column DataFrame
    :param b: A single-column DataFrame
    :return: Estimated memory needed (in MiB)
    """
    cost_a = a.memory_usage(index=True, deep=True).values
    cost_b = b.memory_usage(index=True, deep=True).values
    cost_cols = cost_a[1] * b.shape[0] + cost_b[1] * a.shape[0]

    if pd.__version__ >= '2.0.0' and (isinstance(a.index, pd.RangeIndex)
                                      or isinstance(b.index, pd.RangeIndex)):
        # RangeIndex is used by default in new DataFrames
        cost_idx = 128
    else:
        # Indices are sometimes stored more efficiently, e.g. RangeIndex,
        # so take the max unit cost of either index and multiply by resulting size
        cost_idx = max(a.index.values.itemsize, b.index.values.itemsize) * a.shape[0] * b.shape[0]

    return (cost_cols + cost_idx) / 1024**2


def _validate_input_col_names(on, left_on, right_on) -> tuple:
    if on is None and left_on is None and right_on is None:
        raise KeyError('Column to join on must be specified '
                       '(via "on" or "left_on" and "right_on").')
    left_on = on if on is not None else left_on
    right_on = on if on is not None else right_on
    if isinstance(left_on, list) or isinstance(left_on, tuple):
        raise KeyError('Pandance operation only supports joining on a single column.')
    return left_on, right_on


def _get_join_column_names(left: pd.DataFrame, right: pd.DataFrame, suffixes: tuple) -> list:
    """
    Return only the header (column names) of a join operation.
    """
    merge_cols = []
    right_cols = set(right.columns)
    for col_name in left.columns:
        if col_name in right_cols:
            merge_cols.append(col_name + suffixes[0])
            right_cols -= {col_name}
            right_cols |= {col_name + suffixes[1]}
        else:
            merge_cols.append(col_name)
    for col_name in right_cols:
        merge_cols.append(col_name)
    return merge_cols
