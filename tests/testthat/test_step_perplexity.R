library(sneer)
context("Step Perplexity")

# Test that the cost shows three spikes due to the perplexity change
# but that the optimizer is able to reduce the cost after each spike
# does not change the output kernel precision as perplexity changes
test_that("see three spikes due to perplexity change", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(),
    opt = back_nag(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_step_perp(perplexities = seq(75, 25, length.out = 3),
                                     num_scale_iters = 10, verbose = FALSE,
                                     modify_kernel_fn = NULL),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, verbose = FALSE,
                             report_every = 1),
    max_iter = 20,
    export = c("report")
  )

  expect_equal(ssne_iris$report$costs[,"norm"],
               c(0.936,   0.8311,  0.7047,  0.5821,  0.4717,  # perp 75
                 0.08272, 0.0709,  0.06375, 0.06042, 0.05849, # perp 50
                 0.2278,  0.221,   0.2117,  0.2009,  0.1893,  # perp 25
                 0.1776,  0.1662,  0.1556,  0.146,   0.1375, 0.1301),
               tolerance = 5e-4, scale = 1)
})

test_that("use default kernel adjustment, results should be different", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(),
    opt = back_nag(),
    preprocess = make_preprocess(auto_scale = TRUE),
    init_inp = inp_step_perp(perplexities = seq(75, 25, length.out = 3),
                                     num_scale_iters = 10, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 20,
    export = c("report")
  )
  expect_equal(ssne_iris$report$costs[,"norm"],
               c(0.9291,  0.9261,  0.9219,  0.9171,  0.9119, # perp 75
                 0.9085,  0.9028,  0.8968,  0.8907,  0.8843, # perp 50
                 0.8550,  0.8475,  0.8399,  0.8321,  0.8243, # perp 25
                 0.8164,  0.8084,  0.8003,  0.7921,  0.784,  0.7757),
               tolerance = 5e-4, scale = 1)
})

test_that("first five costs with single perplexity and custom beta", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(beta = 0.006667),
    opt = back_nag(),
    preprocess = make_preprocess(auto_scale = TRUE),
    init_inp = inp_from_perp(perplexity = 75, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 4,
    export = c("report")
  )
  expect_equal(ssne_iris$report$costs[,"norm"],
               c(0.9291,  0.9261,  0.9219,  0.9171,  0.9119),
               tolerance = 5e-4, scale = 1)
})
