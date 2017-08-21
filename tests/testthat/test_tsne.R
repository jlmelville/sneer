library(sneer)
context("t-SNE")

# Run from my fork of Justin Donaldson's R implementation of tsne that can
# be initialized from a PCA scores plot
# tsne(iris[, -5], whiten = FALSE, init_from_PCA = TRUE, epoch = 1, max_iter = 10)
expected_costs <- c(11.2438479963908, 8.35199989895571, 14.7602442989366,
                    14.7064992247826, 14.4988197136326, 14.3769176728282,
                    14.0586777979453, 13.9379958268856, 13.0821973618748,
                    12.1879481102101)

test_that("sneer produces tsne results close to another implementation", {
  iris_rtsne <- sneer(iris, scale_type = "matrix", perp_kernel_fun = "sqrt_exp",
                      eta = 500, init = "p", exaggerate = 4, perplexity = 30,
                      opt = "tsne", max_iter = 9, ret = c("costs"),
                      report_every = 1, plot_type = NULL)
  expect_equal(iris_rtsne$costs[, "cost"], expected_costs, tol = 1e-4)
})
