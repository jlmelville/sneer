library(sneer)
context("Gradients")

# This tests that the analytical gradients match those from a finite difference
# calculation.

preprocess <- make_preprocess(range_scale_matrix = TRUE,
                              verbose = FALSE)
out_init <- out_from_PCA(verbose = FALSE)
opt <- gradient_descent()

gfd <- function(embedder, diff = 1e-4) {
  gradient_fd(embedder$opt, embedder$inp, embedder$out, embedder$method, diff = diff)$gm
}

gan <- function(embedder) {
  gradient(embedder$inp, embedder$out, embedder$method)$gm
}

expect_grad <- function(method,
                        label = "",
                        info = label,
                        inp_init = inp_from_perp(perplexity = 20,
                                                 verbose = FALSE),
                        tolerance = 1e-4) {

  embedder <- init_embed(iris[1:50, 1:4], method,
                         preprocess = preprocess,
                         init_inp = inp_init,
                         init_out = out_init,
                         opt = opt)
  grad_fd <- gfd(embedder)
  grad_an <- gan(embedder)

  expect_equal(grad_an, grad_fd, tolerance = 1e-4, label = label,
               expected.label = "finite difference gradient")
}

test_that("Distance gradients", {
  expect_grad(mmds(), inp_init = NULL, label = "mmds")
  expect_grad(smmds(), inp_init = NULL, label = "smmds")
  expect_grad(sammon_map(), inp_init = NULL, label = "sammon")
})

test_that("SNE gradients", {
  expect_grad(tsne(), label = "tsne")
  expect_grad(ssne(), label = "ssne")
  expect_grad(asne(), label = "asne")
  expect_grad(tasne(), label = "tasne")
})

test_that("Heavy Tailed gradient", {
  expect_grad(hssne(), label = "hssne")
})

test_that("Reverse SNE gradients", {
  expect_grad(rasne(), label = "rasne")
  expect_grad(rssne(), label = "rssne")
  expect_grad(rtsne(), label = "rtsne")
})

test_that("NeRV gradients", {
  expect_grad(nerv(), label = "nerv")
  expect_grad(snerv(), label = "snerv")
  expect_grad(hsnerv(), label = "hsnerv")
})

test_that("NeRV gradients with fixed (or no) bandwidth", {
  expect_grad(unerv(beta = 1), label = "nerv b=1")
  expect_grad(usnerv(beta = 1), label = "snerv b=1")
  expect_grad(uhsnerv(beta = 1), label = "hsnerv b=1")
  expect_grad(tnerv(), label = "tnerv")
})

test_that("JSE gradients", {
  expect_grad(jse(), label = "jse")
  expect_grad(sjse(), label = "sjse")
  expect_grad(hsjse(), label = "hsjse")
})

test_that("Plugin gradients", {
  expect_grad(asne_plugin(), tolerance = 1e-5, label = "plugin asne")
  expect_grad(ssne_plugin(), tolerance = 1e-5, label = "plugin ssne")
  expect_grad(tsne_plugin(), tolerance = 1e-5, label = "plugin tsne")
  expect_grad(hssne_plugin(), tolerance = 1e-5, label = "plugin hssne")
  expect_grad(tasne_plugin(), tolerance = 1e-5, label = "plugin tasne")

  expect_grad(rasne_plugin(), tolerance = 1e-5, label = "plugin rasne")
  expect_grad(rssne_plugin(), tolerance = 1e-5, label = "plugin rssne")
  expect_grad(rtsne_plugin(), tolerance = 1e-5, label = "plugin rtsne")

  expect_grad(unerv_plugin(), tolerance = 1e-5, label = "plugin unerv")
  expect_grad(usnerv_plugin(), tolerance = 1e-5, label = "plugin usnerv")
  expect_grad(uhsnerv_plugin(), tolerance = 1e-5, label = "plugin uhsnerv")
  expect_grad(tnerv_plugin(), tolerance = 1e-5, label = "plugin tnerv")

  expect_grad(nerv_plugin(), tolerance = 1e-5, label = "plugin nerv")
  expect_grad(snerv_plugin(), tolerance = 1e-5, label = "plugin snerv")
  expect_grad(hsnerv_plugin(), tolerance = 1e-5, label = "plugin hsnerv")

  expect_grad(jse_plugin(), tolerance = 1e-5, label = "plugin jse")
  expect_grad(sjse_plugin(), tolerance = 1e-5, label = "plugin sjse")
  expect_grad(hsjse_plugin(), tolerance = 1e-5, label = "plugin hsjse")
})
