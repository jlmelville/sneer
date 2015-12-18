#' sneer: Stochastic Neighbor Embedding Experiments in R.
#'
#' A package for similarity-preserving embedding.
#'
#' @section Examples
#'
#' @examples
#'
#' tsne_iris <- embed_sim(iris[, 1:4], perplexity = 25, method = tsne(),
#'                        preprocess = make_preprocess(
#'                          range_scale_matrix = TRUE),
#'                        init_out = make_init_out(from_PCA = TRUE),
#'                        epoch = make_epoch(plot_func = make_iris_plot()))
#'


#' tsne_iris <- embed_sim(iris[, 1:4], perplexity = 50, method = tsne(), preprocess = make_preprocess(range_scale_matrix = TRUE), init_out = make_init_out(), opt = make_opt(step_size = jacobs(inc_fn = partial(`+`, 0.2), dec_mult = 0.8, min_step_size = 0.1), update = no_momentum(), normalize_grads = FALSE), epoch = make_epoch(plot_func = make_iris_plot()))


#' tsne_iris <- embed_sim(iris[, 1:4], perplexity = 50, method = tsne(),
#' preprocess = make_preprocess(range_scale_matrix = TRUE),
#' init_out = make_init_out(),
#' opt = make_opt(step_size = jacobs(inc_fn = partial(`+`, 0.2),
#' dec_mult = 0.8, min_step_size = 0.1), update = step_momentum(),
#' normalize_grads = FALSE),
#' tricks = make_tricks(early_exaggeration = TRUE),
#' epoch = make_epoch(plot_func = make_iris_plot()))

#'
#' tsne_iris <- prob_embed_f(iris[, 1:4], plot_func = iris_plot(),
#'                           perplexity = 50,
#'                           gradient = tsne(cost_func = kullback_leibler_cost),
#'                           input_weight = sqrt_exp_weight(),
#'                           range_scale_matrix = TRUE,
#'                           Opt = opt(step_size = bold_driver(),
#'                            update = nesterov_non_convex_momentum(),
#'                           gradient = nesterov))
#' Should converge to 0.05539
#'
#' Removed 0 columns, 4 remaining
#' Range Scaling
#' sigma: Min. : 0.1268 |1st Qu. : 0.1462 |Median : 0.1694 |Mean : 0.1717 |3rd Qu. : 0.1998 |Max. : 0.2187 |
#'  beta: Min. : 10.46 |1st Qu. : 12.53 |Median : 17.41 |Mean : 18.3 |3rd Qu. : 23.39 |Max. : 31.12 |
#'  perps: Min. : 50 |1st Qu. : 50 |Median : 50 |Mean : 50 |3rd Qu. : 50 |Max. : 50 |
#'  x2p: Min. : 0 |1st Qu. : 3.138e-05 |Median : 0.0006403 |Mean : 0.006667 |3rd Qu. : 0.008674 |Max. : 0.1286 |
#'  Pjoint: Min. : 2.22e-16 |1st Qu. : 3.994e-07 |Median : 5.542e-06 |Mean : 4.444e-05 |3rd Qu. : 5.913e-05 |Max. : 0.000825 |
#'  Shannon Entropy 8.961 Effective perp: 498.3 Psum 1
#'
#'
#' DEFAULTS: gaussian weight P, range scale, PCA, normalized gradients,
#'  steepest descent
#' + bold driver, no early exaggeration, no momentum tsne_iris <-
#' prob_embed_f(iris[,1:4], plot_func=iris_plot(), perplexity=50,
#' gradient=tsne(),export=c('In'))
#' Range Scaling
#' sigma: Min. : 0.05397 |1st Qu. : 0.06713 |Median : 0.08705 |Mean : 0.09248 |3rd Qu. : 0.1168 |Max. : 0.1493 |
#' beta: Min. : 22.45 |1st Qu. : 36.66 |Median : 65.98 |Mean : 75.7 |3rd Qu. : 111 |Max. : 171.7 |
#' perps: Min. : 50 |1st Qu. : 50 |Median : 50 |Mean : 50 |3rd Qu. : 50 |Max. : 50 |
#' x2p: Min. : 0 |1st Qu. : 5.2e-08 |Median : 0.0001954 |Mean : 0.006667 |3rd Qu. : 0.01084 |Max. : 0.07498 |
#' Pjoint: Min. : 2.22e-16 |1st Qu. : 1.834e-08 |Median : 2.548e-06 |Mean : 4.444e-05 |3rd Qu. : 7.232e-05 |Max. : 0.0004652 |
#' Shannon Entropy 8.967
#' Effective perp: 3921
#' Psum 1
#' PCA: 2 components explained 97.77% variance
#' Epoch: Iteration #0 cost = 0.8882
#' Epoch: Iteration #100 cost = 0.0767
#' Epoch: Iteration #200 cost = 0.07115
#' Epoch: Iteration #300 cost = 0.06898
#' Epoch: Iteration #400 cost = 0.06776
#' Epoch: Iteration #500 cost = 0.06699
#' Epoch: Iteration #600 cost = 0.06646
#' Epoch: Iteration #700 cost = 0.06607
#' Epoch: Iteration #800 cost = 0.06578
#' Epoch: Iteration #900 cost = 0.06557
#' Epoch: Iteration #1000 cost = 0.06541
#' Nbr overlap: Min.: 0.64 |1st Qu. : 0.9 |Median : 0.94 |Mean : 0.9367 |3rd Qu. : 1 |Max. : 1 |
#' Nbr enrichment: Min. : 1.907 |1st Qu. : 2.682 |Median : 2.801 |Mean : 2.791 |3rd Qu. : 2.98 |Max. : 2.98 |
#' rho: Min. : 0.3671 |1st Qu. : 0.9577 |Median : 0.9681 |Mean : 0.9592 |3rd Qu. : 0.9913 |Max. : 0.9982 |
#' wrho: Min. : 0.6963 |1st Qu. : 0.9292 |Median : 0.9509 |Mean : 0.9439 |3rd Qu. : 0.9743 |Max. : 0.9966 |

#' # Momentum and early exaggeration
#' tsne_iris <- prob_embed_f(iris[, 1:4], plot_func = iris_plot(),
#'                           perplexity = 50, gradient = tsne(),
#'                           Opt = opt(update = step_momentum()),
#'                           early_exaggeration = TRUE, export = c('In'))
#'
#' # Nesterov Accelerated Gradient
#' asne_iris <- prob_embed_f(iris[, 1:4], plot_func = iris_plot(),
#'                           perplexity = 50, gradient = asne(),
#'                           Opt = opt(step_size = bold_driver(),
#'                                     update = nesterov_non_convex_momentum(),
#'                                     gradient = nesterov),
#'                           export = c('In'))
#'
#' # JSE s1k
#' jse_s1k <- prob_embed_f(s1k[, 1:9], plot_func = make_plot_func(s1k, 'Label'),
#'                         perplexity = 30,
#'                         gradient = hajse(kappa = 0.5, alpha = 1.5e-8),
#'                         Opt = opt(step_size = bold_driver(),
#'                                   update = nesterov_non_convex_momentum(),
#'                                   gradient = nesterov),
#'                         export=c('In'))
#'
#' mnist hajse_m6k <- prob_embed_f(m1k$x, plot_func = mnist_plot(m1k),
#'                                perplexity = 30,
#'                                gradient = hajse(alpha = 1, beta = 0.5),
#'                                Opt = opt(step_size = bold_driver(),
#'                                          update =
#'                                            nesterov_non_convex_momentum(),
#'                                          gradient = nesterov),
#'                                export = c('In'))
#'
#' global asymmetric probs
#' hhasneg_s1k <- prob_embed_f(s1k[, 1:9],
#'                             plot_func = make_plot_func(s1k, 'Label'),
#'                             perplexity = 50,
#'                             gradient = hhasne(alpha = 1, global = TRUE),
#'                             Opt = opt(step_size = bold_driver(),
#'                                       update =
#'                                        nesterov_non_convex_momentum(),
#'                                       gradient = nesterov),
#'                             export=c('In'))
#'
#' hasne_m1k <- prob_embed_f(m1k$x, plot_func = mnist_plot(m1k), perplexity=50,
#'                           gradient = hhasne(alpha = 1, global = TRUE),
#'                           Opt = opt(step_size = bold_driver(),
#'                                     update = nesterov_non_convex_momentum(),
#'                                     gradient = nesterov),
#'                           export=c('In'), reperplexitize_every = 100)
#'
#' @docType package
#' @name sneer
NULL
