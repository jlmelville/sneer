context("Dynamic HSSNE")

gradient_fd_xi <- function(res, diff = 1e-4, eps = .Machine$double.eps) {

  inp <- res$inp
  out <- res$out
  method <- res$method

  xi_old <- a2x(method$kernel$alpha)

  xi_fwd <- xi_old + diff
  alpha_fwd <- x2a(xi_fwd)
  method$kernel$alpha <- alpha_fwd
  out <- set_solution(inp, out$ym, method, "ym", out)
  cost_fwd <- calculate_cost(method, inp, out)

  xi_back <- xi_old - diff
  alpha_back <- x2a(xi_back)
  method$kernel$alpha <- alpha_back
  out <- set_solution(inp, out$ym, method, "ym", out)
  cost_back <- calculate_cost(method, inp, out)

  (cost_fwd - cost_back) / (xi_fwd - xi_back)
}

a2x <- function(a, eps = .Machine$double.eps) {
  sqrt(a - eps)
}

x2a <- function(x, eps = .Machine$double.eps) {
  x * x + eps
}

embedder <- function(method,
                     inp_init = inp_from_perp(perplexity = 20,
                                              verbose = FALSE)) {
  init_embed(iris[1:50, 1:4], method,
             preprocess = make_preprocess(verbose = FALSE),
             init_inp = inp_init,
             init_out = out_from_PCA(verbose = FALSE),
             opt = mize_grad_descent())
}

test_that("DHSSNE analytical gradient is correct for range of alpha", {
  for (alpha in c(1e-3, 0.25, 0.5, 0.75, 1, 2, 5, 10)) {
    res <- embedder(dhssne(alpha = alpha))
    fd_grad <- gradient_fd_xi(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})
