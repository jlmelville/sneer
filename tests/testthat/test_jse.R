library(sneer)
context("JSE")

test_embed <- function(method) {
  method$verbose <- FALSE
  embed_prob(iris[, 1:4], method = method, max_iter = 50,
             init_inp = inp_from_perp(perplexity = 50, verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = gradient_descent())
}

all_costs <- function(result) {
  result$report$costs[, "cost"]
}

# JSE with kappa -> 0 should be equivalent to ASNE
asne_iris <- test_embed(asne())
jse_iris_k0 <- test_embed(jse(kappa = 0))
test_that("JSE with kappa -> 0 should be equivalent to ASNE", {
expect_equal(all_costs(asne_iris), all_costs(jse_iris_k0), tolerance = 0.001)
})

# JSE with kappa = 1 should be equivalent to rASNE
rasne_iris <- test_embed(rasne())
jse_iris_k1 <- test_embed(jse(kappa = 1))
test_that("JSE with kappa = 1 should be equivalent to RASNE", {
expect_equal(all_costs(rasne_iris), all_costs(jse_iris_k1), tolerance = 0.05)
})

# HSJSE kappa 0 alpha 0 equivalent to SSNE
ssne_iris <- test_embed(ssne())
hsjse_iris_k0a0 <- test_embed(hsjse(kappa = 0, alpha = 0))
test_that("HSJSE kappa 0 alpha 0 equivalent to SSNE", {
expect_equal(all_costs(ssne_iris), all_costs(hsjse_iris_k0a0),
             tolerance = 0.001, scale = 1)
})

# SJSE kappa 0 equivalent to SSNE
sjse_iris_k0 <- test_embed(sjse(kappa = 0))
test_that("SJSE kappa 0 equivalent to SSNE", {
expect_equal(all_costs(ssne_iris), all_costs(sjse_iris_k0),
             tolerance = 0.001, scale = 1)
})

# HSJSE kappa 1 alpha 0 approaches "reverse" SSNE but can't get as close as
# other formulations
rssne_iris <- test_embed(rssne())
hsjse_iris_k1a0 <- test_embed(hsjse(kappa = 1, alpha = 0))
test_that("HSJSE kappa 1 alpha approaches 'reverse' SSNE", {
expect_equal(all_costs(rssne_iris), all_costs(hsjse_iris_k1a0),
             tolerance = 0.005, scale = 1)
})

# HSJSE kappa 0 alpha 1 equivalent to t-SNE
tsne_iris <- test_embed(tsne())
hsjse_iris_k0a1 <- test_embed(hsjse(kappa = 0, alpha = 1))
test_that("HSJSE kappa 0 alpha 1 equivalent to t-SNE", {
expect_equal(all_costs(tsne_iris), all_costs(hsjse_iris_k0a1),
             tolerance = 0.001, scale = 1)
})

# HSJSE kappa 1 alpha 1 equivalent to "reverse" t-SNE but need a lower tolerance
# like reverse SSNE
rtsne_iris <- test_embed(rtsne())
hsjse_iris_k1a1 <- test_embed(hsjse(kappa = 1, alpha = 1))
test_that("HSJSE kappa 1 alpha 1 approached 'reverse' t-SNE", {
expect_equal(all_costs(rtsne_iris), all_costs(hsjse_iris_k1a1),
             tolerance = 0.005, scale = 1)
})
