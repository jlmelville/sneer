context("Dynamic SNE")

inp_df <- iris[1:50, 1:4]
nr <- nrow(inp_df)

gradient_fd_xi <- function(res, param_name = "alpha", diff = 1e-5,
                           eps = .Machine$double.eps) {
  inp <- res$inp
  out <- res$out
  method <- res$method

  xi_old <- a2x(method$kernel[[param_name]])

  xi_fwd <- xi_old + diff
  cost_fwd <- update_and_calc(inp, out, method,param_name, x2a(xi_fwd))

  xi_back <- xi_old - diff
  cost_back <- update_and_calc(inp, out, method, param_name, x2a(xi_back))

  (cost_fwd - cost_back) / (xi_fwd - xi_back)
}

gradient_fd_xi_point <- function(res, param_name = "alpha", diff = 1e-5,
                                 eps = .Machine$double.eps) {
  inp <- res$inp
  out <- res$out
  method <- res$method

  grad <- rep(0, nrow(out$ym))

  for (i in 1:length(grad)) {
    xi_old <- a2x(method$kernel[[param_name]][i])

    xi_fwd <- xi_old + diff
    cost_fwd <- update_and_calc_i(inp, out, method, i, param_name, x2a(xi_fwd))

    xi_back <- xi_old - diff
    cost_back <- update_and_calc_i(inp, out, method, i, param_name,
                                   x2a(xi_back))

    grad[i] <- (cost_fwd - cost_back) / (xi_fwd - xi_back)
  }

  grad
}

update_and_calc <- function(inp, out, method, param_name, param_value) {
  method$kernel[[param_name]] <- param_value
  out <- set_solution(inp, out$ym, method, "ym", out)
  calculate_cost(method, inp, out)
}

update_and_calc_i <- function(inp, out, method, i, param_name, param_value) {
  method$kernel[[param_name]][i] <- param_value
  out <- set_solution(inp, out$ym, method, "ym", out)
  calculate_cost(method, inp, out)
}

a2x <- function(a, eps = .Machine$double.eps) {
  sqrt(a - eps)
}

x2a <- function(x, eps = .Machine$double.eps) {
  x * x + eps
}

embedder <- function(method,
                     inp_init = inp_from_perp(perplexity = 20,
                                              verbose = FALSE),
                     inp_df = iris[1:50, 1:4]) {
  init_embed(inp_df, method,
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

test_that("iHSSNE analytical gradient is correct for range of alpha", {
  for (alpha in c(1e-3, 0.25, 0.5, 0.75, 1, 2, 5, 10)) {
    res <- embedder(ihssne(alpha = seq(alpha, alpha * 2, length.out = nr)))
    fd_grad <- gradient_fd_xi_point(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})

test_that("ht-SNE analytical gradient is correct for range of dof", {
  for (dof in c(1e-3, 0.01, 0.1, 1, 10, 100, 500)) {
    res <- embedder(htsne(dof = dof))
    fd_grad <- gradient_fd_xi(res, param_name = "dof")
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method,
                                   a2x(res$method$kernel$dof))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(dof))
  }
})

test_that("it-SNE analytical gradient is correct for range of dof", {
  for (dof in c(1e-3, 0.01, 0.1, 1, 10, 100, 500)) {
    res <- embedder(itsne(dof = seq(dof, dof * 2, length.out = nr)))
    fd_grad <- gradient_fd_xi_point(res, param_name = "dof")
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method,
                                   a2x(res$method$kernel$dof))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(dof))
  }
})
