library(sneer)
context("Gradients")

# This tests that the analytical gradients match those from a finite difference
# calculation.

preprocess <- make_preprocess(range_scale_matrix = TRUE,  verbose = FALSE)
out_init <- out_from_PCA(verbose = FALSE)
inp_init <- inp_from_perp(perplexity = 20, verbose = FALSE)
inp_aw <- function() { inp_from_perp(perplexity = 45, verbose = FALSE) }
inp_ms <- function() { inp_from_perps_multi(perplexities = seq(45, 25, length.out = 3),
                                num_scale_iters = 0, verbose = FALSE) }
inp_ums <- function() { inp_from_perps_multi(perplexities = seq(45, 25, length.out = 3),
                               num_scale_iters = 0, modify_kernel_fn = NULL,
                               verbose = FALSE) }
inp_tms <- function() { inp_from_perps_multi(perplexities = seq(45, 25, length.out = 3),
                                num_scale_iters = 0,
                                modify_kernel_fn = transfer_kernel_precisions,
                                verbose = FALSE) }

aw <- function(method) {
  lreplace(method,
       inp_updated_fn = nerv_inp_update,
       update_out_fn = make_update_out(keep = c("qm", "wm", "d2m", "qcm"))
  )
}

gfd <- function(embedder, diff = 1e-4) {
  gradient_fd(embedder$inp, embedder$out, embedder$method, diff = diff)$gm
}

gan <- function(embedder) {
  gradient(embedder$inp, embedder$out, embedder$method)$gm
}

# useful for interactive examination of analytical gradients only, diff param is
# ignored, but means you don't have to delete so much when changing a call to
# hgfd
hgan <- function(method, inp_init = inp_from_perp(perplexity = 20,
                                                  verbose = FALSE),
                 diff = 1e-5) {

  embedder <- init_embed(iris[1:50, 1:4], method,
                         preprocess = preprocess,
                         init_inp = inp_init,
                         init_out = out_init,
                         opt = gradient_descent())
  head(gan(embedder))
}

# useful for interactive examination of analytical gradients only
hgfd <- function(method,
                 inp_init = inp_from_perp(perplexity = 20,
                                          verbose = FALSE),
                 diff = 1e-5) {
  embedder <- init_embed(iris[1:50, 1:4], method,
                         preprocess = preprocess,
                         init_inp = inp_init,
                         init_out = out_init,
                         opt = gradient_descent())
  head(gfd(embedder, diff = diff))
}

expect_grad <- function(method,
                        label = "",
                        info = label,
                        inp_init = inp_from_perp(perplexity = 20,
                                                 verbose = FALSE),
                        diff = 1e-5,
                        tolerance = 1e-6,
                        scale = 1) {

  embedder <- init_embed(iris[1:50, 1:4], method,
                         preprocess = preprocess,
                         init_inp = inp_init,
                         init_out = out_init,
                         opt = gradient_descent())
  grad_fd <- gfd(embedder, diff = diff)
  grad_an <- gan(embedder)

  expect_equal(grad_an, grad_fd, tolerance = tolerance, scale = scale,
               label = label, info = info,
               expected.label = "finite difference gradient")
}

test_that("Distance gradients", {
  expect_grad(mmds(), inp_init = NULL, label = "mmds")
  expect_grad(smmds(), inp_init = NULL, label = "smmds")
  expect_grad(sammon_map(), inp_init = NULL, label = "sammon")
})

test_that("SNE gradients", {
  expect_grad(asne(), label = "asne")
  expect_grad(ssne(), label = "ssne")
  expect_grad(tsne(), label = "tsne")
  expect_grad(tasne(), label = "tasne")
})

test_that("Heavy Tailed gradient", {
  expect_grad(hssne(), label = "hssne", diff = 1e-4)
})

test_that("Reverse SNE gradients", {
  expect_grad(rasne(), label = "rasne")
  expect_grad(rssne(), label = "rssne")
  expect_grad(rtsne(), label = "rtsne")
})

test_that("SNE gradients with asymmetric weights", {
  expect_grad(aw(asne()), label = "asne-aw", inp_init = inp_aw())
  expect_grad(aw(rasne()), label = "rasne-aw", inp_init = inp_aw())
})

test_that("NeRV gradients", {
  expect_grad(nerv(), label = "nerv", inp_init = inp_aw())
  expect_grad(snerv(), label = "snerv", inp_init = inp_aw())
  expect_grad(hsnerv(), label = "hsnerv", inp_init = inp_aw())
})

test_that("NeRV gradients with fixed (or no) precision", {
  expect_grad(unerv(beta = 1), label = "unerv b=1")
  expect_grad(usnerv(beta = 1), label = "usnerv b=1")
  expect_grad(uhsnerv(beta = 1), label = "uhsnerv b=1", diff = 1e-4)
  expect_grad(tnerv(), label = "tnerv")
})

test_that("JSE gradients", {
  expect_grad(jse(), label = "jse")
  expect_grad(sjse(), label = "sjse")
  expect_grad(hsjse(), label = "hsjse", diff = 1e-4)
})

test_that("Plugin gradients", {
  expect_grad(asne_plugin(), label = "plugin asne")
  expect_grad(ssne_plugin(), label = "plugin ssne")
  expect_grad(tsne_plugin(), label = "plugin tsne")
  expect_grad(hssne_plugin(), label = "plugin hssne", diff = 1e-4)
  expect_grad(tasne_plugin(), label = "plugin tasne")

  expect_grad(rasne_plugin(), label = "plugin rasne")
  expect_grad(rssne_plugin(), label = "plugin rssne")
  expect_grad(rtsne_plugin(), label = "plugin rtsne")

  expect_grad(unerv_plugin(), label = "plugin unerv")
  expect_grad(usnerv_plugin(), label = "plugin usnerv")
  expect_grad(uhsnerv_plugin(), label = "plugin uhsnerv", diff = 1e-4)
  expect_grad(tnerv_plugin(), label = "plugin tnerv")

  expect_grad(nerv_plugin(), label = "plugin nerv")

  expect_grad(jse_plugin(), label = "plugin jse")
  expect_grad(sjse_plugin(), label = "plugin sjse")
  expect_grad(hsjse_plugin(), label = "plugin hsjse", diff = 1e-4)
})

test_that("Plugin gradients with asymmetric weights", {
  expect_grad(aw(asne_plugin()), label = "plugin asne-aw", inp_init = inp_aw())
  expect_grad(aw(rasne_plugin()), label = "plugin rasne-aw", inp_init = inp_aw())
  expect_grad(aw(ssne_plugin()), label = "plugin ssne-aw", inp_init = inp_aw())
  expect_grad(aw(nerv_plugin()), label = "plugin nerv-aw", inp_init = inp_aw())
  expect_grad(aw(snerv_plugin()), label = "plugin snerv-aw", inp_init = inp_aw())
  expect_grad(aw(jse_plugin()), label = "plugin jse-aw", inp_init = inp_aw())
  expect_grad(aw(sjse_plugin()), label = "plugin sjse-aw", inp_init = inp_aw())
})

test_that("Multiscale gradients", {
  expect_grad(asne_plugin(verbose = FALSE), label = "plugin ms asne",
              inp_init = inp_ms())
  expect_grad(ssne_plugin(verbose = FALSE), label = "plugin ms ssne",
              inp_init = inp_ms())
  expect_grad(rasne_plugin(verbose = FALSE), label = "plugin ms rasne",
              inp_init = inp_ms())
  expect_grad(rssne_plugin(verbose = FALSE), label = "plugin ms rssne",
              inp_init = inp_ms())
  expect_grad(unerv_plugin(verbose = FALSE), label = "plugin ms unerv",
              inp_init = inp_ms())
  expect_grad(usnerv_plugin(verbose = FALSE), label = "plugin ms usnerv",
              inp_init = inp_ms())
  expect_grad(nerv_plugin(verbose = FALSE), label = "plugin ms nerv",
              inp_init = inp_ms())
  expect_grad(snerv_plugin(verbose = FALSE), label = "plugin ms snerv",
              inp_init = inp_ms())
  expect_grad(jse_plugin(verbose = FALSE), label = "plugin ms jse",
              inp_init = inp_ms())
  expect_grad(sjse_plugin(verbose = FALSE), label = "plugin ms sjse",
              inp_init = inp_ms())
  expect_grad(hsjse_plugin(verbose = FALSE), label = "plugin ms hsjse",
              inp_init = inp_ms())

  # don't rescale output precisions
  expect_grad(asne_plugin(verbose = FALSE), label = "plugin ums asne",
              inp_init = inp_ums())
  expect_grad(ssne_plugin(verbose = FALSE), label = "plugin ums ssne",
              inp_init = inp_ums())
  expect_grad(nerv_plugin(verbose = FALSE), label = "plugin ums nerv",
              inp_init = inp_ums())
  expect_grad(snerv_plugin(verbose = FALSE), label = "plugin ums snerv",
              inp_init = inp_ums())
  expect_grad(jse_plugin(verbose = FALSE), label = "plugin ums jse",
              inp_init = inp_ums())
  expect_grad(sjse_plugin(verbose = FALSE), label = "plugin ums sjse",
              inp_init = inp_ums())

  # The ultimate challenge: multiscale and use non-uniform kernel parameters
  expect_grad(asne_plugin(verbose = FALSE), label = "plugin tms asne",
              inp_init = inp_tms())
  expect_grad(ssne_plugin(verbose = FALSE), label = "plugin tms ssne",
              inp_init = inp_tms())
  expect_grad(nerv_plugin(verbose = FALSE), label = "plugin tms nerv",
              inp_init = inp_tms())
  expect_grad(snerv_plugin(verbose = FALSE), label = "plugin tms snerv",
              inp_init = inp_tms())
  expect_grad(jse_plugin(verbose = FALSE), label = "plugin tms jse",
              inp_init = inp_tms())
  expect_grad(sjse_plugin(verbose = FALSE), label = "plugin tms sjse",
              inp_init = inp_tms())
})

test_that("importance weighting", {
  expect_grad(importance_weight(ssne()), label = "wssne")
  expect_grad(importance_weight(ssne_plugin()), label = "plugin wssne")
})
