library(sneer)
context("Perplexity")

presult <- d_to_p_perp_bisect(distance_matrix(range_scale_matrix(iris[, 1:4])),
                              weight_fn = sqrt_exp_weight, perplexity = 50,
                              verbose = FALSE)
sigmas <- 1 / sqrt(presult$beta * 2)

test_that("distribution of sigmas is ok", {
  expect_equal(formatC(min(sigmas)), "0.1268")
  expect_equal(formatC(median(sigmas)), "0.1694")
  expect_equal(formatC(mean(sigmas)), "0.1717")
  expect_equal(formatC(max(sigmas)), "0.2187")
})

test_that("distribution of P is ok", {
  expect_equal(formatC(min(presult$pm)), "2.22e-16")
  expect_equal(formatC(median(presult$pm)), "0.0006403")
  expect_equal(formatC(mean(presult$pm)), "0.006667")
  expect_equal(formatC(max(presult$pm)), "0.1286")
})
