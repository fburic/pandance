library(data.table)

setDTthreads(4)

len_a <- 3000
len_b <- 3000
len_overlap <- 1500

df_a <- data.table(val_a = seq(0, len_a - 1))
df_b <- data.table(val_b = seq(len_a - len_overlap, len_a - len_overlap + len_b - 1))

result <- df_a[df_b, on = .(val_a < val_b)]
