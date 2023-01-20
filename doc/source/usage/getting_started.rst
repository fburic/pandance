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
(inner, outer, cross, left, right) with:

- **fuzzy joins**: Match columns with a tolerance. Supports numerical and datetime values.
- **theta joins**: Allows the user to specify arbitrary matching conditions on
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
The models in one list are rather simple, the others much more complex::

    simple_models:             fancy_models:

      +-------+-------+          +-------+-------+
      | model | score |          | model | score |
      +-------+-------+          +-------+-------+
      | A     | 0.2   |          | M1    | 0.1   |
      | B     | 0.5   |          | M2    | 0.54  |
      | C     | 0.7   |          | M3    | 0.8   |
      | D     | 0.9   |          | M4    | 0.89  |
      +-------+-------+          +-------+-------+

We're interested in finding models that perform essentially the same across the two lists,
and take a score tolerance of 0.05.
This can be easily expressed as a fuzzy join on the score column::

    fuzzy_join(simple_models, fancy_models, on='score', tol=0.05, suffixes=('_s', '_f'))

This gives::

    +---------+---------+---------+---------+
    | model_s | score_s | model_f | score_f |
    +---------+---------+---------+---------+
    | B       | 0.5     | M2      | 0.54    |
    | D       | 0.9     | M4      | 0.89    |
    +---------+---------+---------+---------+

See the :py:meth:`fuzzy_join <pandance.fuzzy_join>` documentation for more details
and examples, including for time series data.

Pandance fuzzy joins are fast and consume little memory, primarily by using
`interval trees <https://github.com/chaimleib/intervaltree>`_
as the underlying data structure and avoiding comparing all entries against all entries.
This makes the implementation faster than typical solutions, like the corresponding
R `fuzzyjoin <https://github.com/dgrtwo/fuzzyjoin/>`_ function,
illustrated here on the same dataset::

    | Implementation                     | time (real) | peak memory |
    |------------------------------------+-------------+-------------|
    | pandance.fuzzy_join()              | 2.485s      |   23.2 MB   |
    |                                    |             |             |
    | fuzzyjoin::difference_inner_join() | 7.461s      | 3099.2 MB   |

.. admonition:: Click to see more technical details about the performance aspect.
    :class: toggle

    By representing the longer column (length *M*) as a tolerance interval tree,
    match lookups can be made in :math:`O(\\log{M})`.

    The R ``fuzzyjoin`` implementation (as of version 0.1.6),
    as well as solutions on e.g. StackOverflow,
    perform the operation (conceptually) as a Cartesian join (comparing all against all,
    so a :math:`O(M)` lookup time for each entry in the shorter column),
    followed by filtering on pairs within the tolerance.

    The dataset used for profiling consists of two tables containing 1e4 numbers sampled
    from two normal distributions (means -2 and 2, respectively, and sd = 1).
    The fuzzy join is performed with a tolerance of 0.1 on these two sets,
    resulting in a sort of fuzzy intersection of the populations.
    (The measurements above include the data generation.)

    .. code-block:: shell

        time python test/performance.py
        valgrind --tool=massif python test/performance.py

    .. code-block:: shell

        time /usr/bin/R --slave --no-save --no-restore --no-site-file --no-environ -f test/fuzzy_perf.R
        # Memory profiling done with RStudio

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
and want to find all pairs in which `keywords` appear as substrings of `phrases`::

    keywords:            phrases:

      +---------+          +----------------------------------------------+
      | keyword |          |                                       phrase |
      +---------+          +----------------------------------------------+
      | a       |          | the quick brown fox jumps over the lazy dog  |
      | the     |          | lorem ipsum dolor                            |
      | xyzzy   |          +----------------------------------------------+
      +---------+

A :math:`\theta`-join can be written with a user-specified match relation
:python:`lambda kw, phrase: kw in phrase` like so::

    dance.theta_join(
        keywords, phrases,
        left_on='keyword', right_on='phrase',
        relation=lambda kw, phrase: kw in phrase
    )

Which results in::

    +---------+---------------------------------------------+
    | keyword |                                      phrase |
    +---------+---------------------------------------------+
    |      a  | the quick brown fox jumps over the lazy dog |
    |    the  | the quick brown fox jumps over the lazy dog |
    +---------+---------------------------------------------+

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
