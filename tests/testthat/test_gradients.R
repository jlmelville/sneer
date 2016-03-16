library(sneer)
context("Gradients")

# This tests that the analytical gradients match those from a finite difference
# calculation.

gfd <- function(embedder, diff = 1e-4) {
  gradient_fd(embedder$opt, embedder$inp, embedder$out, embedder$method, diff = diff)$gm
}

gan <- function(embedder) {
  gradient(embedder$inp, embedder$out, embedder$method)$gm
}

preprocess <- make_preprocess(range_scale_matrix = TRUE,
                              verbose = FALSE)
out_init <- out_from_PCA(verbose = FALSE)
opt <- gradient_descent()

expect_grad <- function(method,
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

  expect_equal(grad_fd, grad_an, tolerance = 1e-4)
}

test_that("Distance gradients", {
  expect_grad(mmds(), inp_init = NULL)
  expect_grad(smmds(), inp_init = NULL)
  expect_grad(sammon_map(), inp_init = NULL)
})

test_that("SNE gradients", {
  expect_grad(tsne())
  expect_grad(ssne())
  expect_grad(asne())
  expect_grad(tasne())
})

test_that("Heavy Tailed gradient", {
  expect_grad(hssne())
})

test_that("Reverse SNE gradients", {
  expect_grad(rasne())
  expect_grad(rssne())
  expect_grad(rtsne())
})

test_that("NeRV gradients", {
  expect_grad(nerv())
  expect_grad(tnerv())
  expect_grad(snerv())
  expect_grad(hsnerv())
})

test_that("JSE gradients", {
  expect_grad(jse())
  expect_grad(sjse())
  expect_grad(hsjse())
})

test_that("Plugin gradients", {
  expect_grad(asne_plugin(), tolerance = 1e-5)
  expect_grad(ssne_plugin(), tolerance = 1e-5)
  expect_grad(tsne_plugin(), tolerance = 1e-5)
  expect_grad(hssne_plugin(), tolerance = 1e-5)
})
