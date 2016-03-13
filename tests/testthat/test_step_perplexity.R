library(sneer)
context("Step Perplexity")

# Test that the cost shows three spikes due to the perplexity change
# but that the optimizer is able to reduce the cost after each spike
tsne_iris <- embed_prob(
  iris[, 1:4],
  method = tsne(),
  opt = back_nag(),
  preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
  init_inp = inp_step_perp(75, 25, num_step_iters = 10, num_steps = 2, verbose = FALSE),
  init_out = out_from_PCA(verbose = FALSE),
  reporter = make_reporter(keep_costs = TRUE, verbose = FALSE, report_every = 1),
  max_iter = 20,
  export = c("report")
)

expect_equal(tsne_iris$report$costs[,"norm"],
             c(0.13991669, 0.12536145, 0.10937834, 0.09590116, 0.08609814, # perp 75
               0.19933190, 0.19237251, 0.18537914, 0.17849920, 0.17178827, # perp 50
               0.35130244, 0.34041000, 0.32899044, 0.31753684, 0.30642772, # perp 25
               0.29590717, 0.28609509, 0.27701780, 0.26864370, 0.26091268,
               0.25375590),
             tolerance = 5e-4, scale = 1)
