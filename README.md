Pandance provides advanced relational operations for
[pandas](https://pandas.pydata.org/) DataFrames,
enabling powerful and efficient joins (aka merges).

## Highlights

Pandance extends the set of standard join operations in pandas
(inner, outer, cross, left, right) with:

- **fuzzy joins**: Match columns with a tolerance
- **[theta joins](https://en.wikipedia.org/wiki/Relational_algebra#%CE%B8-join_and_equijoin)**:
  Allows the user to specify arbitrary matching conditions on which to join

Pandance is designed with performance in mind, aiming to provide fast implementations
whenever possible.

## TODO

- `[0.2.0]` fuzzy join support for DateTime data, as well as arbitrary 
  object types supporting comparison and equality
- `[0.2.0]` inequality join: efficient implementation of non-equijoins using inequalities (<, >)
