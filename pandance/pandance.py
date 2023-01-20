import datetime
import logging
from decimal import Decimal, InvalidOperation
from typing import Callable, Optional, Union

import intervaltree as itree
import numpy as np
import pandas as pd
import psutil

__all__ = ['fuzzy_join', 'theta_join', '_estimate_mem_cost_cartesian']


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
        taking :math:`O((N+M) \\cdot \\log_2{M})` time,
        where *M* is the length of the longest of the two DataFrames,
        and *N* of the shorter one, instead of :math:`O(N \\cdot M)`.

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
    if not index_association:
        return _empty_df(left_on, right_on, suffixes)
    index_assoc_short, index_assoc_long = zip(*index_association)

    # Merge on new index to match order of associated left-right indices
    rows_short = shorter_df.loc[(i for i in index_assoc_short)].reset_index(drop=True)
    rows_long = longer_df.loc[(i for i in index_assoc_long)].reset_index(drop=True)

    # Reflect order of input DataFrames
    if swap_col_order:
        rows_short, rows_long = rows_long, rows_short
    join_result = pd.merge(
        rows_short, rows_long,
        left_index=True, right_index=True, suffixes=suffixes
    )
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
                             interval_tree: itree.IntervalTree) -> list:
    index_assoc = []
    colname = df_col.columns[0]
    df_col.apply(
        lambda row: [index_assoc.append(match)
                     for match in _matching_indices_for_value(row[colname],
                                                              row.name,
                                                              interval_tree)],
        axis='columns'
    )
    return index_assoc


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
               relation: Callable[..., bool],
               on: str = None, left_on: str = None, right_on: str = None,
               suffixes: Optional[tuple] = ('_x', '_y')) -> pd.DataFrame:
    """
    Perform an inner join with a user-specified matching ``relation``.

    A *theta-join* is a join operation in which rows in the join columns
    are matched using an arbitrary
    `relation <https://en.wikipedia.org/wiki/Binary_relation>`_  θ
    that holds between the row entries,
    It generalizes equijoins (where θ = equality).
    See examples below and the
    `Wikipedia article <https://en.wikipedia.org/wiki/Relational_algebra#%CE%B8-join_and_equijoin>`_,
    though in Pandance θ is not limited to the typical set of relations
    {<, <=, =, !=, >=, >}. Rather, the user may specify any boolean-valued function
    as a ``relation``, as described below.

    .. warning::
        This operation is **memory-intensive!**
        Since this is a generic operation for any given `theta` relation,
        it's implemented as a Cartesian product of the two ``on`` columns
        in the input DataFrames,
        followed by a filter on the pairs for which the `theta` relation holds.
        So the memory usage is :math:`O(N \\cdot M)`,
        where `N` and `M` are the respective sizes of the ``on`` columns.

        A warning is logged if the estimated requirement is above 75%
        of available memory and a ``MemoryError`` is raised if the estimate exceeds
        available memory.

    :param left: The left-hand side Pandas DataFrame
    :param right: The right-hand side Pandas DataFrame
    :param relation: a **function** or callable object
        of two parameters ``x``, ``y`` that returns ``True``
        if ``x`` is in that relation with ``y``, else ``False``.
        E.g. ``divides(2, 8) == True``.
    :param on: (Single) column name to join on, passed to ``pandas.merge()``
    :param left_on: (Single) column name to join on in the left DataFrame,
        passed to ``pandas.merge()``
    :param right_on: (Single) column name to join on in the right DataFrame,
        passed to ``pandas.merge()``
    :param suffixes: A length-2 sequence where each element is optionally
        a string indicating the suffix to add to overlapping column names
        in left and right respectively, passed to ``pandas.merge()``
    :return: The *theta*-join of the two DataFrames.


    .. seealso::

        :py:func:`fuzzy_join`
            A special case of θ-join, where θ is :math:`\\approx`.
            It's offered as a separate function since it can be implemented more efficiently.
            Consider using it if you're matching numerical values with a tolerance.

    Examples
    --------

    **Numerical relation**

    We have two tables with numerical entries x and y,
    and we want to find those combinations of x and y that
    represent coordinates on the unit circle.

    >>> import pandas as pd
    >>> import pandance as dance
    >>> horiz_vals = pd.DataFrame([0, 1, -1, 0.5], columns=['x'])
    >>> vert_vals = pd.DataFrame([0, 1, -1, 0.5], columns=['y'])

    Here

    .. math:: \\theta (x, y): x^2 + y^2 - 1 = 0

    >>> import math
    >>> circle_coords = dance.theta_join(
    ...     horiz_vals, vert_vals, left_on='x', right_on='y',
    ...     relation = lambda x, y: math.isclose(x**2 + y**2 - 1, 0))
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
    >>> keywords = pd.DataFrame(['a', 'the', 'xyzzy'], columns=['keyword'])
    >>> phrases = pd.DataFrame([
    ...     'the quick brown fox jumps over the lazy dog',
    ...     'lorem ipsum dolor'
    ... ], columns=['phrase'])

    Here `θ(a, b): a substring of b`.

    >>> hits = dance.theta_join(
    ...     keywords, phrases, left_on='keyword', right_on='phrase',
    ...     relation = lambda kw, phrase: kw in phrase)
    >>> hits
     keyword                                       phrase
    0       a  the quick brown fox jumps over the lazy dog
    1     the  the quick brown fox jumps over the lazy dog


    **Inequality relation**

    We're making a groceries list, and we're balancing macronutrients and costs.

    >>> import pandas as pd
    >>> import pandance as dance
    >>> carb_sources = pd.DataFrame([
    ...     ('rice', 34),
    ...     ('oat flakes', 32)
    ... ], columns=['item', 'price'])
    >>> protein_sources = pd.DataFrame([
    ...     ('lentils', 25),
    ...     ('chickpeas', 38),
    ...     ('soy beans', 48)
    ... ], columns=['item', 'price'])

    We want to stock up on a single carb and protein source,
    but we *want the carbs to cost less than the proteins*.
    This can be expressed as the θ-join below, where

    .. math:: \\theta (x, y): x < y

    >>> possible_shopping_combos = dance.theta_join(
    ...     carb_sources, protein_sources, on='price',
    ...     relation = lambda price_carb, price_prot: price_carb < price_prot,
    ...     suffixes=('_carb', '_prot'))
    >>> possible_shopping_combos
        item_carb  price_carb  item_prot  price_prot
    0        rice          34  chickpeas          38
    1        rice          34  soy beans          48
    2  oat flakes          32  chickpeas          38
    3  oat flakes          32  soy beans          48

    .. tip::
        This type of relation can be implemented more efficiently and will be
        offered as a separate operation in release 0.3.0.
    """
    left_on, right_on = _validate_input_col_names(on, left_on, right_on)

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

    def _safe_relation(x, y) -> bool:
        """Wrapper to guard against known exceptions"""
        try:
            return relation(x, y)
        except InvalidOperation:
            return False

    # Filter on theta
    if left_on == right_on:
        left_on, right_on = left_on + suffixes[0], right_on + suffixes[1]
    result = result[
        result.apply(
            lambda row: _safe_relation(row[left_on], row[right_on]),
            axis='columns'
        )
    ]

    # Get other column items from input DataFrames
    result = pd.merge(
        left.loc[result['index' + suffixes[0]]].reset_index(drop=True),
        right.loc[result['index' + suffixes[1]]].reset_index(drop=True),
        left_index=True, right_index=True, suffixes=suffixes
    )
    return result


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
