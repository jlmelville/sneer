library(sneer)
context("Similarity")

irisd2 <- dist(iris[, 1:4]) ^ 2

expect_equal(heavy_tail_weight(irisd2, alpha = 1.5e-8), exp_weight(irisd2))
expect_equal(heavy_tail_weight(irisd2, alpha = 1.5e-8, beta = 2),
             exp_weight(irisd2, beta = 2))
expect_equal(heavy_tail_weight(irisd2, alpha = 1), tdist_weight(irisd2))
