library(sneer)
context("wSSNE")

do_embed <- function(method, input_weight_fn = exp_weight) {
  method$verbose <- FALSE
  embed_prob(iris[, 1:4], method = method, max_iter = 50,
           init_inp = inp_from_perp(perplexity = 50, verbose = FALSE,
                                    input_weight_fn = input_weight_fn),
           init_out = out_from_PCA(verbose = FALSE),
           preprocess = make_preprocess(verbose = FALSE),
           reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                    verbose = FALSE),
           export = c("report", "inp", "method"),
           verbose = FALSE, opt = gradient_descent())
}

test_that("wSSNE gives different results to SSNE", {
  ssne_embed <- do_embed(ssne())
  wssne_embed <- do_embed(importance_weight(ssne()))
  expect_equal(ssne_embed$cost, 0.0375, tolerance = 1e-4)
  expect_equal(wssne_embed$cost, 0.0495, tolerance = 1e-4)
  # expect warnings with knn and iris because of identical observations, which
  # means certain perplexity values can't be achieved.
  knn_wssne_embed <- do_embed(importance_weight(ssne()),
                              input_weight_fn = step_weight)
  expect_equal(knn_wssne_embed$cost, 0.099, tolerance = 1e-3)

  knn_centrality <- centrality(knn_wssne_embed$inp, knn_wssne_embed$method)
  expect_equal(min(knn_centrality), 62.02, tolerance = 1e-3)
  expect_equal(median(knn_centrality), 99)
  expect_equal(mean(knn_centrality), 100)
  expect_equal(max(knn_centrality), 144.9, tolerance = 1e-3)

  exp_centrality <- centrality(wssne_embed$inp, wssne_embed$method)
  expect_equal(min(exp_centrality), 65.34, tolerance = 1e-3)
  expect_equal(median(exp_centrality), 102.2, tolerance = 1e-3)
  expect_equal(mean(exp_centrality), 100)
  expect_equal(max(exp_centrality), 131, tolerance = 1e-3)
})

test_that("weighting works with row probabilities too", {
  asne_embed <- do_embed(asne())
  expect_equal(asne_embed$cost, 8.74, tolerance = 1e-4)
  wasne_embed <- do_embed(importance_weight(asne()))
  expect_equal(wasne_embed$cost, 6.91, tolerance = 1e-3)
  woasne_embed <- do_embed(importance_weight(asne(),
                                         centrality_fn = outdegree_centrality))
  expect_equal(woasne_embed$cost, asne_embed$cost, tolerance = 1e-6,
              info = "using outdegree centrality should be equivalent to ASNE")
  expect_true(all(Map(formatC, centrality(woasne_embed$inp,
                                          woasne_embed$method)) == 50),
              info = "outdegree centrality is constant with row probabilities")
})
