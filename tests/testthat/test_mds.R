library(sneer)
context("MDS")

# Compared to:
# formatC(smacof::mds(dist(iris[, 1:4]), init = prcomp(iris[, 1:4],
#         center = TRUE, retx = TRUE)$x[, 1:2], eps = 1.5e-8)$stress)
# = "0.03272" - close enough
test_that("mmds is close to smacof::mds result", {

mds_iris <- embed_dist(iris[, 1:4],
                       method = mmds(),
                       opt = mize_bold_nag_adapt(),
                       reporter = make_reporter(
                         extra_costs = c("kruskal_stress"), verbose = FALSE),
                       export = c("report"), verbose = FALSE, max_iter = 40)
expect_equal(mds_iris$report$kruskal_stress, 0.03273, tolerance = 1e-4,
             scale = 1)


  # nest inside so can compare to mds_iris result
  test_that("plugin mmds is close to smacof::mds result", {
    method <- mmds()
    method$stiffness <- distance_stiffness()
    plug_mds_iris <- embed_dist(iris[, 1:4],
                                method = method,
                                opt = mize_bold_nag_adapt(),
                                reporter = make_reporter(
                                  extra_costs = c("kruskal_stress"), verbose = FALSE),
                                export = c("report"), verbose = FALSE, max_iter = 40)
    expect_equal(plug_mds_iris$report$kruskal_stress,
                 mds_iris$report$kruskal_stress)
  })
})



# Compared to:
# formatC(MASS::sammon(dist(iris[c(1:142, 144:150), 1:4]),
#                      y = prcomp(iris[c(1:142, 144:150), 1:4], center = TRUE,
#                      retx = TRUE)$x[, 1:2], tol = 1.5e-8,
#                      trace = FALSE)$stress)
# = "0.004015" - also close enough.
# Reinitializing MASS::sammon with output from sammon_map keeps the lower
# stress configuration sammon_map finds, and also agrees that the initial
# stress is 0.00396.
# Reinitialize sammon_map with the MASS:sammon output reduces the stress back
# down to 0.00396 and also agrees that the initial stress is 0.004015.
# NB MASS::sammon is unable to cope with identical data, so we have to drop
# iris[143,]
test_that("sammon map gives results close to MASS::sammon", {
sammon_iris <- embed_dist(iris[c(1:142, 144:150), 1:4],
                          method = sammon_map(),
                          opt = mize_bold_nag(),
                          reporter = make_reporter(verbose = FALSE),
                          export = c("report"), verbose = FALSE, max_iter = 40)
expect_equal(sammon_iris$report$cost, 0.00396, tolerance = 1e-4, scale = 1)
})

# Nothing to compare the SSTRESS version with, so just characterise it
test_that("SMMDS", {
smds_iris <- embed_dist(iris[, 1:4],
                       method = smmds(),
                       opt = mize_bold_nag_adapt(),
                       reporter = make_reporter(
                         extra_costs = c("kruskal_stress"), verbose = FALSE),
                       export = c("report"), verbose = FALSE, max_iter = 30)
expect_equal(smds_iris$report$kruskal_stress, 0.0364, tolerance = 1e-4,
             scale = 1)
})
