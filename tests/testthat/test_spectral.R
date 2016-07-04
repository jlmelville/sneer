library(sneer)
context("Spectral Direction")

# characterizes the spectral direction method rather than recreates any
# external data
tsne_iris <- embed_prob(iris[, 1:4],
                        method = tsne(verbose = FALSE),
                        opt = optim_spectral(line_search = "bold"),
                        init_inp = inp_from_perp(
                          perplexity = 50,
                          input_weight_fn = sqrt_exp_weight,
                          verbose = FALSE),
                        preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                     verbose = FALSE),
                        max_iter = 200,
                        reporter = make_reporter(verbose = FALSE),
                        export = c("report"),
                        verbose = FALSE)
expect_equal(tsne_iris$report$norm, 0.0526, tolerance = 1.e-4, scale = 1)
