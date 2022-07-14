import logging
import psutil
from typing import Callable, Union, Optional

import pandas as pd

__all__ = ['fuzzy_join', 'theta_join', '_estimate_mem_cost_cartesian']


logger = logging.getLogger()


def fuzzy_join(left: pd.DataFrame, right: pd.DataFrame,
               on: str = None, left_on: str = None, right_on: str = None,
               tol: float = 1e-2,
               suffixes: tuple = ('_x', '_y')) -> pd.DataFrame:
    raise NotImplementedError


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

    The ``fuzzy_join()`` operation is a special case where
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
    result = pd.merge(left[[left_on]], right[[right_on]], how='cross')
    result = result.rename(
        columns={c: i for i, c in enumerate(list(result.columns))}
    )

    # Filter on theta
    result = result[
        result.apply(
            lambda row: relation(row[0], row[1]),
            axis='columns'
        )
    ]

    # Get other column items from input DataFrames
    result = (
        pd.merge(left, result, left_on=left_on, right_on=0)
        .merge(right,
               left_on=1, right_on=right_on, suffixes=suffixes)
    ).drop(
        columns=[0, 1]
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
