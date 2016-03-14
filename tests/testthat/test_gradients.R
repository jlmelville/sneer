library(sneer)
context("Gradients")

# This tests that the analytical gradients match those from a finite difference
# calculation.

preprocess <- make_preprocess(range_scale_matrix = TRUE,
                              verbose = FALSE)
out_init <- out_from_PCA(verbose = FALSE)
opt <- gradient_descent()

expect_grad <- function(method,
                        inp_init = inp_from_perp(perplexity = 20,
                                                 verbose = FALSE)) {
  embedder <- init_embed(iris[1:50, 1:4], method,
                         preprocess = preprocess,
                         init_inp = inp_init,
                         init_out = out_init,
                         opt = opt)
  grad_fd <- gradient_fd(embedder$opt, embedder$inp, embedder$out,
                         embedder$method)$gm
  grad_an <- gradient(embedder$inp, embedder$out, embedder$method)$gm

  expect_equal(grad_fd, grad_an, tolerance = 1e-4)
}

expect_grad(mmds(), inp_init = NULL)
expect_grad(smmds(), inp_init = NULL)
expect_grad(sammon_map(), inp_init = NULL)
expect_grad(tsne())
expect_grad(ssne())
expect_grad(asne())
expect_grad(tasne())
expect_grad(hssne())
expect_grad(rasne())
expect_grad(rssne())
expect_grad(rtsne())
expect_grad(nerv())
expect_grad(tnerv())
expect_grad(snerv())
expect_grad(hsnerv())
expect_grad(jse())
expect_grad(sjse())
expect_grad(hsjse())
