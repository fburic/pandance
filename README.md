# Pandance

A collection of advanced Pandas DataFrame operations:

- fuzzy joins
- the more generic [theta joins](https://en.wikipedia.org/wiki/Relational_algebra#%CE%B8-join_and_equijoin)

As the scope of the package is providing advanced functionality,
it depends on several additional packages besides Pandas.
Also towards this end, the intent is to use newer versions of Pandas.

It is intended as a complement to the 
[Panda Grove](https://panda-grove.readthedocs.io/en/latest/) package 
for managing multiple DataFrames and performing n-ary merges,
which only depends on (slightly older) Pandas and adds minimal API overhead.
