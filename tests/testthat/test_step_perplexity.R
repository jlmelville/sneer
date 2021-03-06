library(sneer)
context("Step Perplexity")

# Test that the cost shows three spikes due to the perplexity change
# but that the optimizer is able to reduce the cost after each spike
# does not change the output kernel precision as perplexity changes
test_that("see three spikes due to perplexity change", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(),
    opt = mize_bold_nag(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_step_perp(perplexities = seq(75, 25, length.out = 3),
                                     num_scale_iters = 10, verbose = FALSE,
                                     modify_kernel_fn = NULL),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, verbose = FALSE,
                             report_every = 1, convergence_iter = 10),
    max_iter = 20,
    export = c("report")
  )

  expect_equal(ssne_iris$report$costs[,"norm"],
               c(0.9359,  0.8310,  0.6974,  0.5616,  0.4354, # perp 75
                 0.0765,  0.0686,  0.0661,  0.0669,  0.0660, # perp 50
                 0.2360,  0.2246,  0.2081,  0.1884,  0.1677, # perp 25
                 0.1485,  0.1328,  0.1208,  0.1119,  0.1054,  0.1024),
               tolerance = 5e-4, scale = 1,
               label = "SSNE step perp no kernel adjustment")
})

test_that("use default kernel adjustment, results should be different", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(),
    opt = mize_bold_nag(),
    preprocess = make_preprocess(auto_scale = TRUE),
    init_inp = inp_from_step_perp(perplexities = seq(75, 25, length.out = 3),
                                     num_scale_iters = 10, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             convergence_iter = 10,
                             verbose = FALSE),
    max_iter = 20,
    export = c("report")
  )
  expect_equal(ssne_iris$report$costs[,"norm"],
               c(0.9291,  0.9261,  0.9216,  0.9162,  0.9098, # perp 75
                 0.9047,  0.8965,  0.8872,  0.8768,  0.8651, # perp 50
                 0.8265,  0.8106,  0.7932,  0.7741,  0.7534, # perp 25
                 0.7309,  0.7066,  0.6806,  0.6529,  0.6235,  0.5926),
               tolerance = 5e-4, scale = 1, label = "SSNE step perp")
})

test_that("first five costs with single perplexity and custom beta", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(beta = 0.006667),
    opt = mize_back_nag(),
    preprocess = make_preprocess(auto_scale = TRUE),
    init_inp = inp_from_perp(perplexity = 75, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 4,
    export = c("report")
  )
  expect_equal(ssne_iris$report$costs[,"norm"],
               c(0.9291,  0.9261,  0.9219,  0.915,  0.905),
               tolerance = 5e-4, scale = 1,
               label = "SSNE with single perp but custom beta")
})
