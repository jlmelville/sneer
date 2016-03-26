library(sneer)
context("Gradients")

# This tests that the analytical gradients match those from a finite difference
# calculation.

preprocess <- make_preprocess(range_scale_matrix = TRUE,
                              verbose = FALSE)
out_init <- out_from_PCA(verbose = FALSE)
inp_init <- inp_from_perp(perplexity = 20,
                          verbose = FALSE)
inp_aw <- inp_from_perp(perplexity = 45,
              verbose = FALSE)

aw <- function(method) {
  lreplace(method,
           inp_updated_fn = transfer_kernel_bandwidths,
           update_out_fn = make_update_out(keep = c("qm", "wm", "d2m", "qcm"))
)
}

gfd <- function(embedder, diff = 1e-4) {
  gradient_fd(embedder$inp, embedder$out, embedder$method, diff = diff)$gm
}

gan <- function(embedder) {
  gradient(embedder$inp, embedder$out, embedder$method)$gm
}

expect_grad <- function(method,
                        label = "",
                        info = label,
                        inp_init = inp_from_perp(perplexity = 20,
                                                 verbose = FALSE),
                        diff = 1e-4,
                        tolerance = 1e-4) {

  embedder <- init_embed(iris[1:50, 1:4], method,
                         preprocess = preprocess,
                         init_inp = inp_init,
                         init_out = out_init,
                         opt = gradient_descent())
  grad_fd <- gfd(embedder, diff = diff)
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
  expect_grad(asne(), label = "asne")
  expect_grad(ssne(), label = "ssne")
  expect_grad(tsne(), label = "tsne")
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

test_that("SNE gradients with asymmetric weights", {
  expect_grad(aw(asne()), label = "asne-aw", inp_init = inp_aw)
  expect_grad(aw(rasne()), label = "rasne-aw", inp_init = inp_aw)
  expect_grad(aw(asne_plugin()), label = "plugin asne-aw", inp_init = inp_aw)
  expect_grad(aw(rasne_plugin()), label = "plugin rasne-aw", inp_init = inp_aw)
})

test_that("NeRV gradients", {
  expect_grad(nerv(), label = "nerv", inp_init = inp_aw)
  expect_grad(snerv(), label = "snerv", inp_init = inp_aw)
  expect_grad(hsnerv(), label = "hsnerv", inp_init = inp_aw)
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
  expect_grad(asne_plugin(), label = "plugin asne")
  expect_grad(ssne_plugin(), label = "plugin ssne")
  expect_grad(tsne_plugin(), label = "plugin tsne")
  expect_grad(hssne_plugin(), label = "plugin hssne")
  expect_grad(tasne_plugin(), label = "plugin tasne")

  expect_grad(rasne_plugin(), label = "plugin rasne")
  expect_grad(rssne_plugin(), label = "plugin rssne")
  expect_grad(rtsne_plugin(), label = "plugin rtsne")

  expect_grad(unerv_plugin(), label = "plugin unerv")
  expect_grad(usnerv_plugin(), label = "plugin usnerv")
  expect_grad(uhsnerv_plugin(), label = "plugin uhsnerv")
  expect_grad(tnerv_plugin(), label = "plugin tnerv")

  expect_grad(nerv_plugin(), label = "plugin nerv")

  expect_grad(jse_plugin(), label = "plugin jse")
  expect_grad(sjse_plugin(), label = "plugin sjse")
  expect_grad(hsjse_plugin(), label = "plugin hsjse")
})
