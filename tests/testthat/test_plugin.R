library(sneer)
context("Plugin")

embed_with <- function(method, modify_kernel_fn = NULL) {
  embed_prob(iris[1:50, 1:4],
             method = method,
             init_inp = inp_from_perp(
               perplexity = 15,
               modify_kernel_fn = modify_kernel_fn,
               input_weight_fn = sqrt_exp_weight,
               verbose = FALSE),
             preprocess = make_preprocess(range_scale_matrix = TRUE,
                                          verbose = FALSE),
             max_iter = 20,
             reporter = make_reporter(verbose = FALSE),
             export = c("report", "method"),
             verbose = FALSE,
             opt = mize_bold_nag_adapt())
}

expect_plugin_equal <- function(method_name, tolerance = .Machine$double.eps,
                                modify_kernel_fn = NULL) {
  method1 <- get(method_name)()
  method1$stiffness <- plugin_stiffness()
  method1$verbose <- FALSE

  method2 <- get(method_name)()
  method2$verbose <- FALSE

  expect_same_method(method1, method2, tolerance = tolerance,
                     info = method_name, modify_kernel_fn = modify_kernel_fn)
}

expect_same_method <- function(method1, method2,
                               tolerance = .Machine$double.eps, info = NULL,
                               scale = 1,
                               modify_kernel_fn = NULL) {
  embed1 <- embed_with(method1, modify_kernel_fn = modify_kernel_fn)
  embed2 <- embed_with(method2, modify_kernel_fn = modify_kernel_fn)

  expect_match(embed1$method$stiffness$name, "Plugin", info = info)
  expect_true(embed2$method$stiffness$name != "Plugin", info = info)

  expect_equal(embed1$cost, embed2$cost, tolerance = tolerance, info = info,
               scale = 1)
}

test_that("plugin method results match non-plugin methods", {
  expect_plugin_equal("asne", tolerance = 1e-4)
  expect_plugin_equal("ssne", tolerance = 1e-5)
  expect_plugin_equal("tsne", tolerance = 1e-5)
  expect_plugin_equal("tasne", tolerance = 1e-5)
  expect_plugin_equal("hssne", tolerance = 1e-5)
  expect_plugin_equal("rasne", tolerance = 1e-5)
  expect_plugin_equal("rssne", tolerance = 1e-5)
  expect_plugin_equal("rtsne", tolerance = 1e-5)
  expect_plugin_equal("nerv", tolerance = 1e-4)
  expect_plugin_equal("snerv", tolerance = 1e-5)
  expect_plugin_equal("hsnerv", tolerance = 1e-5)
  expect_plugin_equal("tnerv", tolerance = 1e-5)
  expect_plugin_equal("nerv", tolerance = 1e-5,
                      modify_kernel_fn = transfer_kernel_precisions)
  expect_plugin_equal("jse", tolerance = 1e-5)
  expect_plugin_equal("sjse", tolerance = 1e-5)
  expect_plugin_equal("hsjse", tolerance = 1e-5)
  expect_plugin_equal("tpsne", tolerance = 1e-5)
})

