# library(sneer)
# context("MSJSE")
#
# msjse_iris <-
#   embed_prob(iris[, 1:4], method = jse(verbose = FALSE), max_iter = 20,
#            init_inp = inp_multiscale(num_scale_iters = 16, verbose = FALSE),
#            init_out = out_from_PCA(verbose = FALSE),
#            preprocess = make_preprocess(verbose = FALSE),
#            reporter = make_reporter(report_every = 1, keep_costs = TRUE,
#                                     verbose = FALSE),
#            export = c("report"), verbose = FALSE, opt = gradient_descent())
#
# expect_equal(msjse_iris$report$cost, 91.28975, tolerance = 1e-4, scale = 1)
#
# expect_equal(msjse_iris$report$costs[,"norm"],
#              c(0.3397878, 0.3095258, 0.2780233, 0.2459909, # perp 32
#                0.3927030, 0.3600638, 0.3266617, 0.2936029, # perp 16
#                0.4191758, 0.3871961, 0.3554713, 0.3243794, # perp 8
#                0.4296219, 0.3998572, 0.3702789, 0.3410493, # perp 4
#                0.4290268, 0.4014187, 0.3740687, 0.3469663, 0.3200241 # perp 2
#                ),
#              tolerance = 5e-4, scale = 1)
#
