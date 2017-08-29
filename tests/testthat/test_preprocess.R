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

test_that("tsne-scaling", {
  iris_ns <- make_preprocess(tsne = TRUE, verbose = FALSE)(iris[, -5])
  expect_equal(unname(apply(iris_ns$xm, 2, mean)), rep(0, 4))

  # the relative standard deviations are preserved
  raw_sd_ratio <- max(apply(iris[, -5], 2, sd)) / min(apply(iris[, -5], 2, sd))
  norm_sd_ratio <- max(apply(iris_ns$xm, 2, sd)) / min(apply(iris_ns$xm, 2, sd))
  expect_equal(raw_sd_ratio, norm_sd_ratio)
})

# Test passing list
test_that("column range scaling", {
  scaled <- make_preprocess(scale_list = list(col = "range"), verbose = FALSE)(iris[ , -5])
  expect_equal(unname(apply(scaled$xm, 2, min)), rep(0, 4))
  expect_equal(unname(apply(scaled$xm, 2, max)), rep(1, 4))
})

test_that("matrix range scaling", {
  scaled <- make_preprocess(scale_list = list(mat = "range"), verbose = FALSE)(iris[ , -5])
  expect_equal(max(scaled$xm), 1)
  expect_equal(min(scaled$xm), 0)
  # column max and mins should NOT be 0 and 1
  expect_false(isTRUE(all.equal(unname(apply(scaled$xm, 2, min)), rep(0, 4))))
  expect_false(isTRUE(all.equal(unname(apply(scaled$xm, 2, max)), rep(1, 4))))
})

test_that("column sd scaling", {
  scaled <- make_preprocess(scale_list = list(col = "sd"), verbose = FALSE)(iris[ , -5])
  expect_equal(unname(apply(scaled$xm, 2, mean)), rep(0, 4))
  expect_equal(unname(apply(scaled$xm, 2, sd)), rep(1, 4))
})

test_that("matrix sd scaling", {
  scaled <- make_preprocess(scale_list = list(mat = "sd"), verbose = FALSE)(iris[ , -5])
  expect_equal(mean(scaled$xm), 0)
  expect_equal(sd(scaled$xm), 1)
  # columns should not have been scaled to mean 0, sd 1
  expect_false(isTRUE(all.equal(unname(apply(scaled$xm, 2, mean)), rep(0, 4))))
  expect_false(isTRUE(all.equal(unname(apply(scaled$xm, 2, sd)), rep(1, 4))))
})

test_that("column max scaling", {
  scaled <- make_preprocess(scale_list = list(col = "max"), verbose = FALSE)(iris[ , -5])
  expect_equal(unname(apply(scaled$xm, 2, function(x) { max(abs(x)) })),
               rep(1, 4))
})

test_that("matrix max scaling", {
  scaled <- make_preprocess(scale_list = list(mat = "max"), verbose = FALSE)(iris[ , -5])
  expect_equal(max(abs(scaled$xm)), 1)
  # Columns should not all be max = 1
  expect_false(isTRUE(all.equal(
    unname(apply(scaled$xm, 2, function(x) { max(abs(x)) })),
    rep(1, 4))))
})

test_that("range scale m then center c", {
  scaled <- make_preprocess(scale_list = list(mat = "range", col = "center"),
                            verbose = FALSE)(iris[, -5])
  expect_equal(unname(apply(scaled$xm, 2, mean)), rep(0, 4))

  # the relative standard deviations are preserved
  raw_sd_ratio <- max(apply(iris[, -5], 2, sd)) / min(apply(iris[, -5], 2, sd))
  norm_sd_ratio <- max(apply(scaled$xm, 2, sd)) / min(apply(scaled$xm, 2, sd))
  expect_equal(raw_sd_ratio, norm_sd_ratio)
})

test_that("center c then range scale m (list order matters)", {
  scaled <- make_preprocess(scale_list = list(col = "center", mat = "range"),
                            verbose = FALSE)(iris[ , -5])
  expect_equal(max(scaled$xm), 1)
  expect_equal(min(scaled$xm), 0)
  # column max and mins should NOT be 0 and 1
  expect_false(isTRUE(all.equal(unname(apply(scaled$xm, 2, min)), rep(0, 4))))
  expect_false(isTRUE(all.equal(unname(apply(scaled$xm, 2, max)), rep(1, 4))))
})

test_that("center c then max scale m", {
  scaled <- make_preprocess(scale_list = list(col = "center", mat = "max"), verbose = FALSE)(iris[ , -5])
  # means should still be 0
  expect_equal(unname(apply(scaled$xm, 2, mean)), rep(0, 4))

  # max abs value is 1
  expect_equal(abs(max(scaled$xm)), 1)

  # the relative standard deviations are preserved
  raw_sd_ratio <- max(apply(iris[, -5], 2, sd)) / min(apply(iris[, -5], 2, sd))
  norm_sd_ratio <- max(apply(scaled$xm, 2, sd)) / min(apply(scaled$xm, 2, sd))
  expect_equal(raw_sd_ratio, norm_sd_ratio)
})
