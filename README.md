<div align="center" >
  <img src="doc/source/img/pandance_logo.svg" width="250px"><br>
</div>

<h1 align="center">Pandance</h1>

-----------------

Pandance provides advanced relational operations for
[pandas](https://pandas.pydata.org/) DataFrames,
enabling powerful and efficient joins (aka merges).

## Highlights

Pandance extends the set of standard join operations in pandas
(inner, outer, cross, left, right) with:

- **fuzzy joins**: Match columns with a tolerance. Supports numerical and datetime values.
- **inequality join**: Match one column's values that are less / greater than the other column's values.
- **[theta joins](https://en.wikipedia.org/wiki/Relational_algebra#%CE%B8-join_and_equijoin)**:
  Allows the user to specify arbitrary matching conditions on which to join

Pandance is designed with performance in mind, aiming to provide fast implementations
whenever possible.

## Installation

```shell
pip install pandance
```

## Usage

See the [documentation](https://pandance.readthedocs.io)
