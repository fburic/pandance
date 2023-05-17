.. role:: python(code)
  :language: python
  :class: highlight

.. _getting_started:

Getting Started
===============

Installation
------------

Pandance is available from PyPI

.. code-block:: shell

   pip install pandance


Basic Usage
-----------

The Pandance operations are meant to integrate as seamlessly as possible with
`pandas <https://pandas.pydata.org>`_, and have a very similar API, e.g.

.. code-block:: python

    import pandance as dance

    result_df = dance.fuzzy_join(df_a, df_b, on='column_name', tol=0.05)

Each operation has examples of usage in its documentation.
The Highlights section below showcases some of Pandance's functionality.


Highlights
----------

Pandance extends the set of standard join operations in pandas
(inner, outer, cross, left, right) with the operations:

- **fuzzy join**: Match columns with a tolerance. Supports numerical and datetime values.
- **inequality join**: Match one column's values that are less / greater than the other column's values.
  Supports any type that implements comparisons (numerical, string, datetime, user-defined, etc.).
- **theta join**: Allows the user to specify arbitrary matching conditions on
  which to join.

Pandance is designed with performance in mind, aiming to provide fast implementations
whenever possible.


Fuzzy Joins
"""""""""""

While the most common type of join matches column values exactly,
one will invariably come across situations where
some sort of tolerance must be considered,
for instance matching on timestamps to find events that occur at approximately the same time,
or joining on some type of real-valued (float) score to find entries with similar performance.

One approach is to transform the input DataFrames by binning (discretizing) these values.
A more natural approach would be to consider fuzzy matches.

Taking the previous example of matching things based on some real value,
say we have two sets of model performance scores (0..1).
The models in one list are rather simple, the others much more complex:

.. table::
    :class: container-table
    :width: 35%

    +---------------------+----+-------------------------+
    | ``simple_models``   |    | ``fancy_models``        |
    |                     |    |                         |
    | .. table::          |    | .. table::              |
    |                     |    |                         |
    |    =====   =====    |    |    =====   =====        |
    |    model   score    |    |    model   score        |
    |    =====   =====    |    |    =====   =====        |
    |    A       0.2      |    |    M1      0.1          |
    |    B       0.5      |    |    M2      0.54         |
    |    C       0.7      |    |    M3      0.8          |
    |    D       0.9      |    |    M4      0.89         |
    |    =====   =====    |    |    =====   =====        |
    |                     |    |                         |
    +---------------------+----+-------------------------+

We're interested in finding models that perform essentially the same across the two lists,
and take a score tolerance of 0.05.
This can be easily expressed as a fuzzy join on the score column::

    fuzzy_join(simple_models, fancy_models, on='score', tol=0.05, suffixes=('_s', '_f'))

This gives:

.. table::
    :width: 25%

    =======   =======   =======   =======
    model_s   score_s   model_f   score_f
    =======   =======   =======   =======
    B         0.5       M2        0.54
    D         0.9       M4        0.89
    =======   =======   =======   =======

See the :py:meth:`fuzzy_join <pandance.fuzzy_join>` documentation for more details
and examples, including for time series data.

Performance
~~~~~~~~~~~

Pandance fuzzy joins are fast and consume little memory, primarily by using
`interval trees <https://github.com/chaimleib/intervaltree>`_
as the underlying data structure and avoiding comparing all entries against all entries.
This makes the implementation faster than typical solutions, like the corresponding
R `fuzzyjoin <https://github.com/dgrtwo/fuzzyjoin/>`_ function.
This is illustrated here by the performance on the same dataset consisting of
two overlapping normal distributions (10 000 samples from each, mean=-2 and 2,
respectively, sd=1) with a tolerance of 0.1, which resulted in
106 776 matches.

.. table::
    :width: 50%
    :widths: 25 15 25

    +------------------------------------+-----------+-------------+
    | Implementation                     | time [s]  | memory [MB] |
    +====================================+===========+=============+
    | pandance.fuzzy_join()              |  1.88     |    16.5     |
    +------------------------------------+-----------+-------------+
    | fuzzyjoin::difference_inner_join() |  4.24     |   3070      |
    +------------------------------------+-----------+-------------+

.. admonition:: Click to see more technical details about the performance aspect.
    :class: toggle

    By representing the longer column (length *m*) as a tolerance interval tree,
    match lookups can be made in :math:`O(\log{m})`.

    The R ``fuzzyjoin`` implementation (as of version 0.1.6),
    as well as solutions on e.g. StackOverflow,
    perform the operation as a Cartesian join (comparing all against all,
    so a :math:`O(m)` lookup time for each entry in the shorter column),
    followed by filtering on pairs within the tolerance.

    The dataset used for profiling consists of two tables containing 1e4 numbers sampled
    from two normal distributions (means -2 and 2, respectively, and sd = 1).
    The fuzzy join is performed with a tolerance of 0.1 on these two sets,
    resulting in a sort of fuzzy intersection of the populations.
    Only the join operation time and memory are shown.

    **Pandance (Python) measurement**:

    .. code-block:: shell

        # Speed
        python test/performance.py
        snakeviz $(ls -1rt | tail -n 1)

        # Memory
        # (decorated fuzzy_speed_random with @profile)
        python -m memory_profiler test/performance.py

    **fuzzyjoin (R) measurement**: Used RStudio's profiling
    (``profvis`` package under the hood)

    The profiling scripts are available in the Pandance
    `repo <https://github.com/fburic/pandance/tree/main/test>`_.
    The profiling was performed on a machine with 4x Intel Core i7-8550U @ 4 GHz and 8 GB RAM.
    Pandance is currently single-threaded.


Inequality Joins
""""""""""""""""

Sometimes we want to match table entries based on how they compare.
For example, if we have weather data for two regions A and B,
we may want to find those entries where monthly rainfall was higher in
region B. Assuming for the sake of the example we have two dataframes
``weather_region_a`` and ``weather_region_b`` (a single one may be split by region),
we can use an inequality join to match the two based on
``weather_region_a['rainfall'] < weather_region_b['rainfall']``.

Pandance provides :py:meth:`ineq_join <pandance.ineq_join>` for this operation,
which would look like::

    ineq_join(weather_region_a, weather_region_b, on='rainfall', how='<')

As an example of working with temporal data, say we want to find connecting flights
between locations A and C with a layover in location B.
We have a table with flights from A to B, and another from B to C:

.. table::
    :class: container-table
    :width: 70%

    +----------------------------------------+----+----------------------------------------+
    | ``flights_ab``                         |    | ``flights_bc``                         |
    |                                        |    |                                        |
    | .. table::                             |    | .. table::                             |
    |                                        |    |                                        |
    |    ================  ================  |    |    ================  ================  |
    |    dep               arr               |    |    dep               arr               |
    |    ================  ================  |    |    ================  ================  |
    |    2023-01-01 08:00  2023-01-01 10:00  |    |    2023-01-01 09:00  2023-01-01 12:00  |
    |    2023-01-01 12:00  2023-01-01 14:00  |    |    2023-01-01 14:00  2023-01-01 17:00  |
    |    2023-01-01 16:00  2023-01-01 18:00  |    |    2023-01-01 18:00  2023-01-01 21:00  |
    |    2023-01-01 20:00  2023-01-01 22:00  |    |    2023-01-01 21:00  2023-01-02 00:00  |
    |    ================  ================  |    |    ================  ================  |
    |                                        |    |                                        |
    +----------------------------------------+----+----------------------------------------+

Valid connections are those flights that arrive in B before a departing flight from B.
So we can get those flights with::

    ineq_join(flights_ab, flights_bc, left_on='arr', right_on='dep', how='<',
              suffixes=('_ab', '_bc'))

which gives us a list of connections:

.. table::
    :width: 60%

    ================   ================   ================   ================
    dep_ab             arr_ab             dep_bc             arr_bc
    ================   ================   ================   ================
    2023-01-01 08:00   2023-01-01 10:00   2023-01-01 14:00   2023-01-01 17:00
    2023-01-01 08:00   2023-01-01 10:00   2023-01-01 18:00   2023-01-01 21:00
    2023-01-01 12:00   2023-01-01 14:00   2023-01-01 18:00   2023-01-01 21:00
    2023-01-01 08:00   2023-01-01 10:00   2023-01-01 21:00   2023-01-02 00:00
    2023-01-01 12:00   2023-01-01 14:00   2023-01-01 21:00   2023-01-02 00:00
    2023-01-01 16:00   2023-01-01 18:00   2023-01-01 21:00   2023-01-02 00:00
    ================   ================   ================   ================

A more realistic example would likely entail adding a buffer value,
which can just be done by passing a mutated version of ``flights_ab`` with
the time buffer added to the arrival time column.
Or, if the tables are small, we could use a :py:meth:`theta_join <pandance.theta_join>`
to express the more complex condition.

Since ``ineq_join`` relies on comparisons, any Python object that supports these
may be used for the join columns.
Let's take an example with strings, which in Python are ordered
`lexicographically <https://en.wikipedia.org/wiki/Lexicographic_order>`_.
Suppose we have a small sample of strings and want to find all strings that are
sorted lower in a large database of strings (which here is constructed randomly)::

    query = pd.DataFrame(['bbb', 'ccc'], columns=['string'])

    database = pd.DataFrame(
        [''.join(random.choices(string.ascii_lowercase, k=3)) for _ in range(10)],
        columns=['string']
    )

    ineq_join(query, database, how='>', on='string', suffixes=('_query', '_db'))

In this case, the random database only has a few strings of lower ordering than our query:

.. table::
    :width: 20%

    ============  =========
    string_query  string_db
    ============  =========
    bbb           afn
    ccc           afn
    bbb           afq
    ccc           afq
    ============  =========


Performance
~~~~~~~~~~~

The ``ineq_join`` operation is typically faster than the common straightforward approach
using a Cartesian (cross) join followed by filtering on the inequality condition,
this latter method having the downside of comparing everything with everything,
even if the number of actual matches is much lower.
See the documentation of :py:meth:`ineq_join <pandance.ineq_join>` for more details.

Fro much better performance however, the excellent R
`data.table <https://r-datatable.com>`_ package is recommended, as it supports
inequality joins (which it implements in a similar fashion to ``ineq_join``)
but is two orders of magnitude faster, as shown below on the same dataset.

.. table::
    :width: 65%
    :widths: 60 15 25

    +--------------------------------------------------------+-----------+-------------+
    | Implementation                                         | time [s]  | memory [MB] |
    +========================================================+===========+=============+
    | ``pandance.ineq_join``                                 | 9.26      | 247         |
    +--------------------------------------------------------+-----------+-------------+
    | ``pandance.theta_join`` (cross join with ineq. filter) | 244       | 1000        |
    +--------------------------------------------------------+-----------+-------------+
    | ``data.table`` join with inequality (``1 thread``)     | 0.2       | 30          |
    +--------------------------------------------------------+-----------+-------------+
    | ``data.table`` join with inequality (``4 threads``)    | 0.15      | 60          |
    +--------------------------------------------------------+-----------+-------------+


.. admonition:: Click to see more technical details about the performance measurement.
    :class: toggle

    The benchmark consisted of 2 dataframes *A* and *B* containing increasing integer sequences.
    The two sequences have a parametrized overlap, otherwise A values are smaller than B.
    On a ``<`` join without any overlap, the result is a Cartesian join,
    with :math:`A \cdot B` elements.
    With a nonzero overlap L, the number of matches (rows in the result) is
    :math:`A \cdot B - L^2 + \binom{L}{2}`.
    For the results below *A* = *B* = 3000 and *L* = 1500,
    meaning a result of length 7 874 250.
    Only the join operation time and memory are shown.
    ``data.table`` version ``1.14.8`` was used.

    **Pandance (Python) measurement**:

    .. code-block:: shell

        # Speed
        python test/performance.py
        snakeviz $(ls -1rt | tail -n 1)

        # Memory
        # (decorated ineq_join_overlap_cartesian with @profile)
        python -m memory_profiler test/performance.py

    **data.table (R) measurement**: Used RStudio's profiling
    (``profvis`` package under the hood)

    The profiling scripts are available in the Pandance
    `repo <https://github.com/fburic/pandance/tree/main/test>`_.
    The profiling was performed on a machine with 4x Intel Core i7-8550U @ 4 GHz and 8 GB RAM.
    Pandance is currently single-threaded.


Theta Joins
"""""""""""

While joins naturally capture exact or approximate matching between columns,
in principle pairs of values may be considered to match based on any criteria.

In `relational algebra <https://en.wikipedia.org/wiki/Relational_algebra#%CE%B8-join_and_equijoin>`_,
a :math:`\theta`-join is a join where pairs :math:`(a, b)`
of values from columns A and B are considered to match
if they fulfill a relation :math:`\theta`, which we could write :math:`\theta(a, b) = True`.

Pandance implements an (inner) :py:meth:`theta_join <pandance.theta_join>` that takes a
user-specified boolean-valued function which judges whether pairs of elements match.
This is a departure from the typical limited choice of :math:`\theta`
as an inequality `{<, <=, =, !=, >=, >}`.

For instance, if we have the following tables of strings
and want to find all pairs in which `keywords` appear as substrings of `phrases`:

.. table::
    :class: container-table
    :width: 70%

    +-----------------+----+-----------------------------------------------------+
    | ``keywords``    |    | ``phrases``                                         |
    |                 |    |                                                     |
    | .. table::      |    | .. table::                                          |
    |                 |    |                                                     |
    |    +---------+  |    |    +---------------------------------------------+  |
    |    | keyword |  |    |    | phrase                                      |  |
    |    +=========+  |    |    +=============================================+  |
    |    | a       |  |    |    | the quick brown fox jumps over the lazy dog |  |
    |    +---------+  |    |    +---------------------------------------------+  |
    |    | the     |  |    |    | lorem ipsum dolor                           |  |
    |    +---------+  |    |    +---------------------------------------------+  |
    |    | xyzzy   |  |    |                                                     |
    |    +---------+  |    |                                                     |
    |                 |    |                                                     |
    +-----------------+----+-----------------------------------------------------+

A :math:`\theta`-join can be written with a user-specified match relation
:python:`lambda kw, phrase: kw in phrase` like so::

    dance.theta_join(
        keywords, phrases,
        left_on='keyword', right_on='phrase',
        relation=lambda kw, phrase: kw in phrase
    )

Which results in:

.. table::
    :width: 45%

    =======  ===========================================
    keyword  phrase
    =======  ===========================================
    a        the quick brown fox jumps over the lazy dog
    the      the quick brown fox jumps over the lazy dog
    =======  ===========================================

See the :py:meth:`theta_join <pandance.theta_join>` documentation for more details
and examples.

.. warning::

    Since this Pandance operation allows any user-specified matching relation,
    there is no way of avoiding a Cartesian join of the two join columns
    (comparing everything with everything).
    This will likely consume all available memory for large data sets,
    so care must be taken (although Pandance will warn you first).

    Consider instead using the special case provided by
    :py:meth:`fuzzy_join <pandance.fuzzy_join>`
    whenever possible.
