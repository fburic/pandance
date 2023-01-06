from decimal import Decimal, InvalidOperation
import logging
import psutil
from typing import Callable, Optional, Union

import intervaltree as itree
import numpy as np
import pandas as pd

__all__ = ['fuzzy_join', 'theta_join', '_estimate_mem_cost_cartesian']


logger = logging.getLogger()


def fuzzy_join(left: pd.DataFrame, right: pd.DataFrame,
               on: str = None, left_on: str = None, right_on: str = None,
               tol: Union[float, Decimal] = 1e-3,
               suffixes: tuple = ('_x', '_y')) -> pd.DataFrame:
    """
    Perform an approximate inner join of two DataFrames, on a numerical column.
    E.g. ``1.03 ~= 1``, given an absolute tolerance ``tol = 0.5``.
    The tolerance is inclusive, meaning ``(a - b) <= tol`` is considered a match.

    The joined DataFrame contains both numerical columns that were used in the join.

    .. Warning::

        The matching may misbehave if values are very large and the tolerance small,
        due to the simple absolute tolerance test and floating point representation
        limitations (see *Notes*).

    .. Warning::

        `NaN` and `Inf` values in the joining column will (silently) not yield matches,
        as per the `IEEE 754 <https://en.wikipedia.org/wiki/NaN#Comparison_with_NaN>`_
        standard implemented by NumPy.

    .. Note::

        This operation is a more efficient implementation
        compared to the generic `theta_join <#pandance.theta_join>`_,
        taking *O((N+M) log2 M)* time,
        where *M* is the length of the longest of the two DataFrames,
        and *N* of the other, instead of *O(N*M)*.

    :param left: The left-hand side Pandas DataFrame
    :param right: The right-hand side Pandas DataFrame
    :param on: (Single) numerical column name to join on
    :param left_on: (Single) numerical column name to join on in the left DataFrame
    :param right_on: (Single) numerical column name to join on in the right DataFrame
    :param tol: Numerical tolerance for the fuzzy matching.
    :param suffixes: A length-2 sequence where each element is optionally
        a string indicating the suffix to add to overlapping column names
        in left and right respectively
    :return: The fuzzy join of the two DataFrames.

    Example
    -------

    Given two datasets recording the observation times (0..1) of events,
    perform a fuzzy join on the time column,
    to get only the events that occur at approximately the same time between sets::

      df_a:                               df_b:

          | event    |  time_obs   |          | event    |  time_obs   |
          |----------|-------------|          |----------|-------------|
          | event1   | 0.2         |          | event5   | 0.1         |
          | event2   | 0.5         |          | event6   | 0.54        |
          | event3   | 0.7         |          | event7   | 0.8         |
          | event4   | 0.9         |          | event9   | 0.89        |

    The operation::

        fuzzy_join(df_a, df_b, on='time_obs', tol=0.05, suffixes=('_a', '_b'))

    gives::

        | event_a  |  time_obs_a | event_b  | time_obs_b |
        |----------|-------------|----------|------------|
        | event2   | 0.5         | event6   | 0.54       |
        | event4   | 0.9         | event9   | 0.89       |


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
    if on is None and left_on is None and right_on is None:
        raise KeyError('Column to join on must be specified '
                       '(via "on" or "left_on" and "right_on").')
    left_on = on if not None else left_on
    right_on = on if not None else right_on
    if isinstance(left_on, list) or isinstance(left_on, tuple):
        raise KeyError('Fuzzy join only supports joining on a single column.')

    left, right = _def_validate_and_clean_inputs_to_fuzzy(left, right, left_on, right_on)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return pd.DataFrame([], columns=[left_on + suffixes[0], right_on + suffixes[1]])

    if left.shape[0] <= right.shape[0]:
        shorter_col, longer_col = left_on, right_on
        shorter_df, longer_df = left, right
    else:
        longer_col, shorter_col = left_on, right_on
        longer_df, shorter_df = left, right

    epsilon = np.finfo(np.float32).eps
    if isinstance(left[left_on].values[0], Decimal):
        tol = Decimal(tol)
        epsilon = Decimal(epsilon.item())
    interval_tree = _build_interval_tree(longer_df[[longer_col]], tol, epsilon)

    index_association = _get_fuzzy_match_indices(shorter_df[[shorter_col]], interval_tree)
    if not index_association:
        return pd.DataFrame([], columns=[left_on + suffixes[0], right_on + suffixes[1]])
    index_assoc_short, index_assoc_long = zip(*index_association)

    # Merge on new index to match order of associated left-right indices
    rows_short = shorter_df.loc[(i for i in index_assoc_short)].reset_index(drop=True)
    rows_long = longer_df.loc[(i for i in index_assoc_long)].reset_index(drop=True)
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


def _def_validate_and_clean_inputs_to_fuzzy(left: pd.DataFrame,
                                            right: pd.DataFrame,
                                            left_on: str,
                                            right_on: str) -> tuple:
    supported_dtypes = ['i', 'u', 'f']
    exception_msg = f'Operation only supports joining on columns ' \
                    f'of NumPy types: {supported_dtypes} or decimal.Decimal'

    if left[left_on].dtype.kind not in supported_dtypes:
        if left.shape[0] > 0:
            if not isinstance(left[left_on].values[0], Decimal):
                raise TypeError('Left DataFrame invalid: ' + exception_msg)

    if right[right_on].dtype.kind not in supported_dtypes:
        if right.shape[0] > 0:
            if not isinstance(right[right_on].values[0], Decimal):
                raise TypeError('Right DataFrame invalid: ' + exception_msg)

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


def theta_join(left: pd.DataFrame, right: pd.DataFrame,
               relation: Union[Callable[..., bool], pd.DataFrame],
               on: str = None, left_on: str = None, right_on: str = None,
               suffixes: Optional[tuple] = ('_x', '_y')) -> pd.DataFrame:
    """
    A *theta-join* is a join operation in which entry pairs in two columns
    are matched using an arbitrary
    `relation <https://en.wikipedia.org/wiki/Binary_relation>`_ `theta`
    that holds between these entries,
    as a generalization of equijoins, where this relation is simply equality.
    See examples below and the
    `Wikipedia article <https://en.wikipedia.org/wiki/Relational_algebra#%CE%B8-join_and_equijoin>`_,
    though note that in Pandance `theta` isn't limited to the typical set
    {<, <=, =, !=, >=, >}. Rather, the user may specify arbitrary relations,
    as described below.

    .. Note::
        Because this operation accepts arbitrary relations,
        *reflexivity (i.e. x R x, for all x) is not implemented*.
        A simple example is the "<" relation,
        where reflexivity (*x < x*) doesn't hold.
        So the user must specify reflexivity in the given ``relation`` themselves.

    The `fuzzy_join <#pandance.fuzzy_join>`_ operation is a special case where
    `theta` = `approximately_equal`, offered as a separate function
    since it can be implemented more efficiently.

    .. Warning::
        This operation is **memory-intensive!**
        Since this is a generic operation for any given `theta` relation,
        it's implemented as a Cartesian product of the two ``on`` columns
        in the input DataFrames,
        followed by a filter on the pairs for which the `theta` relation holds.
        So the memory usage is `O(N * M)`,
        where `N` and `M` are the respective sizes of the ``on`` columns.

        A warning is logged if the estimated requirement is above 75%
        of available memory and a ``MemoryError`` is raised if the estimate exceeds
        available memory.

    **Example 1**

    We're merging two inventory tables which unfortunately use alternative
    (synonymous) words for the same items.
    E.g. "drink" in one table and "beverage" in the other.

    Thus, we have the `synonymous` relation between multiple items, i.e.
    a function ``synonymous('drink', 'beverage') == True``,
    while ``synonymous('drink', 'sandwich') == False``,
    This could even be stored in another table of pairs and passed to the
    ``theta_join()`` operation, in which case only items in this table would be
    considered to be in the `synonymous` relation, e.g.::

        | item_a   |  item_b   |
        |----------|-----------|
        | drink    | beverage  |
        | starter  | appetizer |

    So we can perform the theta-join operation in two steps.
    First, provide the synonymous relation:::

        def synonymous(item_a: str, item_b: str) -> bool:
            # logic matching item_a with item_b

    or::

        synonymous = pd.read_csv('item_aliases.csv')

    then call ``theta_join()``::

        theta_join(
            old_inventory, new_inventory,
            relation = synonymous,
            on = 'item'
        )

    **Example 2**

    We have two tables with numerical entries,
    and we want to match those numbers that are equal modulo 64
    (i.e. have the same remainder when dividing by 64).::

        def mod_64(x: int, y: int) -> bool:
            return x % 64 == y % 64 == 0

        theta_join(
            number_set_a, number_set_b,
            relation = mod_64,
            on = 'numeric_value'
        )

    :param left: The left-hand side Pandas DataFrame
    :param right: The right-hand side Pandas DataFrame
    :param relation: a **function** of two parameters ``x``, ``y``
        that returns ``True`` if ``x`` is in that relation with ``y``,
        else ``False``.
        E.g. ``divides(2, 8) == True`` or
        ``synonymous('drink', 'beverage') == True``.
        Alternatively, a **DataFrame** with two columns of item pairs that
        are in the relation (the column names are irrelevant),
        implying all other combinations are not in the relation.

    :param on: (Single) column name to join on, passed to ``pandas.merge()``
    :param left_on: (Single) column name to join on in the left DataFrame,
        passed to ``pandas.merge()``
    :param right_on: (Single) column name to join on in the right DataFrame,
        passed to ``pandas.merge()``
    :param suffixes: A length-2 sequence where each element is optionally
        a string indicating the suffix to add to overlapping column names
        in left and right respectively, passed to ``pandas.merge()``
    :return: The *theta*-join of the two DataFrames.
    """
    left_on = on if not None else left_on
    right_on = on if not None else right_on

    est_mem = _estimate_mem_cost_cartesian(left[[left_on]], right[[right_on]])
    avail_mem = psutil.virtual_memory()
    avail_mem = (avail_mem.total - avail_mem.used) / 1024**2
    if est_mem > avail_mem:
        logger.error(f'The operation requires more memory than is currently available: {est_mem}')
        raise MemoryError
    if est_mem / avail_mem > 0.75:
        logger.warning(f'The operation requires over 75% ({est_mem}) of available memory')

    if isinstance(relation, pd.DataFrame):
        relation_pairs = dict(relation.values)

        def relation(a, b):
            # Explicit check for presence of `a` to avoid any weird issues
            # with None, N/As, and default values
            if a not in relation_pairs:
                # Ought to be the more common case with larger tables
                return False
            else:
                return relation_pairs[a] == b

    # Cartesian join
    result = pd.merge(left[[left_on]].reset_index(),
                      right[[right_on]].reset_index(),
                      how='cross',
                      suffixes=suffixes)

    # Filter on theta
    def _safe_relation(x, y) -> bool:
        """Guard against known exceptions"""
        try:
            return relation(x, y)
        except InvalidOperation:
            return False

    result = result[
        result.apply(
            lambda row: _safe_relation(row[left_on + suffixes[0]],
                                       row[right_on + suffixes[1]]),
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
