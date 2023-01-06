library(fuzzyjoin)
library(tidyverse)

a <- tibble(val = rnorm(10000, mean=-2, sd=1)) %>% mutate(idx = row_number())
b <- tibble(val = rnorm(10000, mean=2, sd=1)) %>% mutate(idx = row_number())

x <- fuzzyjoin::difference_inner_join(
  a, b, by="val", max_dist = 0.1
)
