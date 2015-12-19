library(sneer)
context("SNE")

tsne_iris <- embed_sim(iris[, 1:4],
                       method = tsne_stiffness(),
                       init_inp = make_init_inp(perplexity = 50,
                                                input_weight_fn =
                                                  sqrt_exp_weight,
                                                verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 200,
                       epoch = make_epoch(verbose = FALSE),
                       export = c("epoch_result"),
                       verbose = FALSE)
expect_equal(formatC(tsne_iris$epoch_result$stress), "0.0578")

ssne_iris <- embed_sim(iris[, 1:4],
                       method = ssne_stiffness(),
                       init_inp = make_init_inp(perplexity = 50,
                                                input_weight_fn =
                                                  sqrt_exp_weight,
                                                verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 200,
                       epoch = make_epoch(verbose = FALSE),
                       export = c("epoch_result"),
                       verbose = FALSE)
expect_equal(formatC(ssne_iris$epoch_result$stress), "0.07265")

asne_iris <- embed_sim(iris[, 1:4],
                       method = asne_stiffness(),
                       init_inp = make_init_inp(perplexity = 50,
                                                input_weight_fn =
                                                  sqrt_exp_weight,
                                                verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 200,
                       epoch = make_epoch(verbose = FALSE),
                       export = c("epoch_result"),
                       verbose = FALSE)
expect_equal(formatC(asne_iris$epoch_result$stress), "0.08479")

pca_iris <- embed_sim(iris[, 1:4],
                      method = tsne_stiffness(),
                      init_inp = make_init_inp(perplexity = 25,
                                               input_weight_fn =
                                                 sqrt_exp_weight,
                                               verbose = FALSE),
                      preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                   verbose = FALSE),
                      max_iter = 0, verbose = FALSE,
                      epoch = make_epoch(epoch_every = 1, verbose = FALSE),
                      export = c("epoch_result"))
expect_equal(formatC(pca_iris$epoch_result$cost), "1.598")

context("jacobs")
tsne_iris_jacobs <- embed_sim(iris[, 1:4],
                              method = tsne_stiffness(),
                              init_inp = make_init_inp(perplexity = 25,
                                                       input_weight_fn =
                                                         sqrt_exp_weight,
                                                       verbose = FALSE),
                              preprocess = make_preprocess(
                                range_scale_matrix = TRUE,
                                verbose = FALSE),
                              max_iter = 10, opt =
                                make_opt(
                                  step_size = jacobs(inc_fn = partial(`+`, 0.2),
                                    dec_mult = 0.8, min_step_size = 0.1),
                                  normalize_grads = FALSE),
                              verbose = FALSE,
                              epoch = make_epoch(epoch_every = 1,
                                                 save_costs = TRUE,
                                                 verbose = FALSE),
                              export = c("epoch_result"))
jacobs_costs <- tsne_iris_jacobs$epoch_result$costs[,"cost"]
expect_equal(formatC(jacobs_costs[1]),  "1.598")
expect_equal(formatC(jacobs_costs[2]),  "1.594")
expect_equal(formatC(jacobs_costs[3]),  "1.589")
expect_equal(formatC(jacobs_costs[4]),  "1.584")
expect_equal(formatC(jacobs_costs[5]),  "1.577")
expect_equal(formatC(jacobs_costs[6]),  "1.57")
expect_equal(formatC(jacobs_costs[7]),  "1.562")
expect_equal(formatC(jacobs_costs[8]),  "1.553")
expect_equal(formatC(jacobs_costs[9]),  "1.543")
expect_equal(formatC(jacobs_costs[10]), "1.532")
