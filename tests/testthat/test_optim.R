library(sneer)
context("CG, BFGS and L-BFGS")

test_that("CG with MMDS", {
mds_iris <- embed_dist(iris[, 1:4],
                       method = mmds(),
                       opt = mizer_opt("CG", c2 = 0.1),
                       reporter = make_reporter(
                         extra_costs = c("kruskal_stress"), verbose = FALSE),
                       export = c("report"), verbose = FALSE, max_iter = 40)
expect_equal(mds_iris$report$kruskal_stress, 0.03273, tolerance = 1e-4,
             scale = 1)
})

test_that("BFGS with MMDS", {
  mds_iris <- embed_dist(iris[, 1:4],
                         method = mmds(),
                         opt = mizer_opt("BFGS", c2 = 0.9, scale_hess = TRUE),
                         reporter = make_reporter(
                           extra_costs = c("kruskal_stress"), verbose = FALSE),
                         export = c("report"), verbose = FALSE, max_iter = 40)
  expect_equal(mds_iris$report$kruskal_stress, 0.03273, tolerance = 1e-4,
               scale = 1)
})

test_that("L-BFGS with MMDS", {
  mds_iris <- embed_dist(iris[, 1:4],
                         method = mmds(),
                         opt = mizer_opt("L-BFGS", c2 = 0.9, scale_hess = TRUE,
                                         memory = 5),
                         reporter = make_reporter(
                           extra_costs = c("kruskal_stress"), verbose = FALSE),
                         export = c("report"), verbose = FALSE, max_iter = 40)
  expect_equal(mds_iris$report$kruskal_stress, 0.03273, tolerance = 1e-4,
               scale = 1)
})
