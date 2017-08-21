library(sneer)
context("Preprocess")

test_that("column range scaling", {
  iris_crs <- make_preprocess(range_scale = TRUE, verbose = FALSE)(iris[ , -5])
  expect_equal(unname(apply(iris_crs$xm, 2, min)), rep(0, 4))
  expect_equal(unname(apply(iris_crs$xm, 2, max)), rep(1, 4))
})

test_that("matrix range scaling", {
  iris_mrs <- make_preprocess(range_scale_matrix = TRUE, verbose = FALSE)(iris[ , -5])
  expect_equal(max(iris_mrs$xm), 1)
  expect_equal(min(iris_mrs$xm), 0)
})

test_that("auto scaling", {
  iris_as <- make_preprocess(auto_scale = TRUE, verbose = FALSE)(iris[, -5])
  expect_equal(unname(apply(iris_as$xm, 2, mean)), rep(0, 4))
  expect_equal(unname(apply(iris_as$xm, 2, sd)), rep(1, 4))
})

test_that("normalize", {
  iris_ns <- make_preprocess(normalize = TRUE, verbose = FALSE)(iris[, -5])
  expect_equal(unname(apply(iris_ns$xm, 2, mean)), rep(0, 4))

  # the relative standard deviations are preserved
  raw_sd_ratio <- max(apply(iris[, -5], 2, sd)) / min(apply(iris[, -5], 2, sd))
  norm_sd_ratio <- max(apply(iris_ns$xm, 2, sd)) / min(apply(iris_ns$xm, 2, sd))
  expect_equal(raw_sd_ratio, norm_sd_ratio)
})
