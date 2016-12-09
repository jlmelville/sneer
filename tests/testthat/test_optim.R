library(sneer)
context("Roptim")

# CG does not perform well on this at all: Polak-Ribiere and Beale-Sorensen
# being particularly miserable
test_that("Can run optimize with optim L-BFGS-B", {
tsne_iris <- embed_prob(iris[, 1:4],
                        method = tsne(verbose = FALSE),
                        opt = ropt(method = "L-BFGS-B", batch_iter = 20),
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
expect_equal(tsne_iris$report$norm, 0.0522, tolerance = 1.e-4, scale = 1)
})

# CG does fine on this case, though
test_that("Can run optim CG with MMDS", {
mds_iris <- embed_dist(iris[, 1:4],
                       method = mmds(),
                       opt = ropt(method = "CG", type = 2, batch_iter = 20),
                       reporter = make_reporter(
                         extra_costs = c("kruskal_stress"), verbose = FALSE),
                       export = c("report"), verbose = FALSE, max_iter = 40)
expect_equal(mds_iris$report$kruskal_stress, 0.03273, tolerance = 1e-4,
             scale = 1)
})
