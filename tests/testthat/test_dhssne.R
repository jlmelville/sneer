context("Dynamic SNE")

inp_df <- iris[1:50, 1:4]
nr <- nrow(inp_df)
betas <- seq(1e-3, 1, length.out = nr)
alphas <- c(1e-3, 0.25, 1, 2, 10)

gradient_fd_xi <- function(res, param_names = c("alpha"), diff = 1e-5,
                           eps = .Machine$double.eps) {
  inp <- res$inp
  out <- res$out
  method <- res$method

  gr <- rep(0, length(param_names))
  for (i in 1:length(param_names)) {
    param_name <- param_names[i]
    xi_old <- a2x(method$kernel[[param_name]])

    xi_fwd <- xi_old + diff
    cost_fwd <- update_and_calc(inp, out, method, param_name, x2a(xi_fwd))

    xi_back <- xi_old - diff
    cost_back <- update_and_calc(inp, out, method, param_name, x2a(xi_back))

    gr[i] <- (cost_fwd - cost_back) / (xi_fwd - xi_back)
  }
  gr
}

gradient_fd_xi_point <- function(res, param_names = c("alpha"), diff = 1e-5,
                                 eps = .Machine$double.eps) {
  inp <- res$inp
  out <- res$out
  method <- res$method

  grad <- rep(0, nrow(out$ym) * length(param_names))

  gi <- 0
  for (j in 1:length(param_names)) {
    param_name <- param_names[j]
    for (i in 1:length(method$kernel[[param_name]])) {
      xi_old <- a2x(method$kernel[[param_name]][i])

      xi_fwd <- xi_old + diff
      cost_fwd <- update_and_calc_i(inp, out, method, i, param_name, x2a(xi_fwd))

      xi_back <- xi_old - diff
      cost_back <- update_and_calc_i(inp, out, method, i, param_name,
                                     x2a(xi_back))
      gi <- gi + 1
      grad[gi] <- (cost_fwd - cost_back) / (xi_fwd - xi_back)
    }
  }

  grad
}

update_and_calc <- function(inp, out, method, param_name, param_value) {
  method$kernel[[param_name]] <- param_value
  res <- set_solution(inp, out$ym, method, "ym", out)
  calculate_cost(method, res$inp, res$out)
}

update_and_calc_i <- function(inp, out, method, i, param_name, param_value) {
  method$kernel[[param_name]][i] <- param_value
  res <- set_solution(inp, out$ym, method, "ym", out)
  calculate_cost(method, res$inp, res$out)
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

quick_embed <- function(method, df = iris[, 1:4],
                        opt = mize_grad_descent(),
                        max_iter = 5, report_every = ceiling(max_iter / 10)) {
  embed_prob(df, method = method, max_iter = max_iter,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 1, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), opt = opt)
}

# Test Global Alpha ------------------------------------------------------------

# Symmetric
test_that("DHSSNE analytical gradient is correct for range of single alpha and single beta", {
  for (alpha in alphas) {
    for (beta in c(1, 0.5)) {
      res <- embedder(dhssne(alpha = alpha, beta = beta,
                             xi_eps = .Machine$double.eps, verbose = FALSE))
      fd_grad <- gradient_fd_xi(res)
      an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                     a2x(res$method$kernel$alpha))
      expect_equal(an_grad, fd_grad, tol = 1e-6,
                   info = paste0(formatC(alpha), " ", formatC(beta)))
    }
  }
})

test_that("DHSSNE analytical gradient is correct for range of single alpha and heterogeneous beta", {
  for (alpha in alphas) {
    res <- embedder(dhssne(alpha = alpha, beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6,
                 info = paste0(formatC(alpha)))
  }
})

# Semi Symmetric
test_that("DH3SNE analytical gradient is correct for range of single alpha and single beta", {
  for (alpha in alphas) {
    for (beta in c(1, 0.5)) {
      res <- embedder(dh3sne(alpha = alpha, beta = beta,
                             xi_eps = .Machine$double.eps, verbose = FALSE))
      fd_grad <- gradient_fd_xi(res)
      an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                     a2x(res$method$kernel$alpha))
      expect_equal(an_grad, fd_grad, tol = 1e-6,
                   info = paste0(formatC(alpha), " ", formatC(beta)))
    }
  }
})

test_that("DH3SNE analytical gradient is correct for range of single alpha and heterogeneous beta", {
  for (alpha in alphas) {
    res <- embedder(dh3sne(alpha = alpha, beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))

    fd_grad <- gradient_fd_xi(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})

# Pair-wise
test_that("DHPSNE analytical gradient is correct for range of single alpha and heterogeneous beta", {
  for (alpha in alphas) {
    for (beta in c(1, 0.5)) {
      res <- embedder(dhpsne(alpha = alpha, beta = beta,
                             xi_eps = .Machine$double.eps, verbose = FALSE))
      fd_grad <- gradient_fd_xi(res)
      an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                     a2x(res$method$kernel$alpha))
      expect_equal(an_grad, fd_grad, tol = 1e-6,
                   info = paste0(formatC(alpha), " ", formatC(beta)))
    }
  }
})

test_that("DHPSNE analytical gradient is correct for range of single alpha and heterogeneous beta", {
  for (alpha in alphas) {
    res <- embedder(dhpsne(alpha = alpha, beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))

    fd_grad <- gradient_fd_xi(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})

test_that("DHASNE analytical gradient is correct for range of single alpha and heterogeneous beta", {
  for (alpha in alphas) {
    res <- embedder(dhasne(alpha = alpha, beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))

    fd_grad <- gradient_fd_xi(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})


# Test Point-wise Alphas -------------------------------------------------------

# Symmetric
test_that("iHSSNE analytical gradient is correct for range of multi alpha and multi beta", {
  for (alpha in alphas) {
    res <- embedder(ihssne(alpha = seq(alpha, alpha * 2, length.out = nr),
                           beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi_point(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})

test_that("iHSSNE analytical gradient is correct using generic parameter gradient", {
  for (alpha in alphas) {
    method <- ihssne(alpha = seq(alpha, alpha * 2, length.out = nr),
                     beta = betas,
                     xi_eps = .Machine$double.eps, verbose = FALSE)
    method$gr_alpha <- heavy_tail_cost_gr_alpha_plugin
    method$gr_beta <- heavy_tail_cost_gr_beta_plugin
    res <- embedder(method)
    fd_grad <- gradient_fd_xi_point(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})


# Semi-symmetric
test_that("iH3SNE analytical gradient is correct for range of multi alpha and multi beta", {
  for (alpha in alphas) {
    res <- embedder(ih3sne(alpha = seq(alpha, alpha * 2, length.out = nr),
                           beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi_point(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})

# Pair-wise
test_that("iHPSNE analytical gradient is correct for range of multi alpha and multi beta", {
  for (alpha in alphas) {
    res <- embedder(ihpsne(alpha = seq(alpha, alpha * 2, length.out = nr),
                           beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi_point(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})


test_that("iHASNE analytical gradient is correct for range of multi alpha and multi beta", {
  for (alpha in alphas) {
    res <- embedder(ihasne(alpha = seq(alpha, alpha * 2, length.out = nr),
                           beta = betas,
                           xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi_point(res)
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$alpha))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})

# Doubly Dynamic ----------------------------------------------------------

test_that("DDHSSNE analytical gradient is correct for range of single alpha and single beta", {
  for (alpha in alphas) {
    for (beta in c(1, 0.5)) {
      res <- embedder(ddhssne(alpha = alpha, beta = beta,
                              xi_eps = .Machine$double.eps, verbose = FALSE))
      fd_grad <- gradient_fd_xi(res, param_names = c("alpha", "beta"))
      an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                     a2x(c(res$method$kernel$alpha, res$method$kernel$beta)))
      expect_equal(an_grad, fd_grad, tol = 1e-6,
                   info = paste0(formatC(alpha), " ", formatC(beta)))
    }
  }
})

# "doubly" inhomogeneous: alpha and beta
test_that("DiHSSNE analytical gradient is correct for range of multi alpha and multi beta", {
  for (alpha in alphas) {
    res <- embedder(dihssne(alpha = seq(alpha, alpha * 2, length.out = nr),
                            beta = betas,
                            xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi_point(res, param_names = c("alpha", "beta"))
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(c(res$method$kernel$alpha, res$method$kernel$beta)))

    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})

test_that("DiHASNE analytical gradient is correct for range of multi alpha and multi beta", {
  for (alpha in alphas) {
    res <- embedder(dihasne(alpha = seq(alpha, alpha * 2, length.out = nr),
                            beta = betas,
                            xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi_point(res, param_names = c("alpha", "beta"))
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(c(res$method$kernel$alpha, res$method$kernel$beta)))

    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(alpha))
  }
})


# Dynamic ASNE (Precision Only) -------------------------------------------

test_that("iASNE analytical gradient is correct for non-uniform beta", {
  res <- embedder(iasne(beta = betas,
                        xi_eps = .Machine$double.eps, verbose = FALSE))
  fd_grad <- gradient_fd_xi_point(res, param_names = c("beta"))
  an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                 a2x(res$method$kernel$beta))
  expect_equal(an_grad, fd_grad, tol = 1e-6)
})

test_that("iSSNE analytical gradient is correct for non-uniform beta", {
  res <- embedder(issne(beta = betas,
                        xi_eps = .Machine$double.eps, verbose = FALSE))
  fd_grad <- gradient_fd_xi_point(res, param_names = c("beta"))
  an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                 a2x(res$method$kernel$beta))
  expect_equal(an_grad, fd_grad, tol = 1e-6)
})

test_that("iSSNE works with generic parameter gradient too", {
  method <- lreplace(
    issne(beta = betas, xi_eps = .Machine$double.eps, verbose = FALSE),
    gr_beta = exp_cost_gr_param_plugin
  )
  res <- embedder(method)
  fd_grad <- gradient_fd_xi_point(res, param_names = c("beta"))
  an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                 a2x(res$method$kernel$beta))
  expect_equal(an_grad, fd_grad, tol = 1e-6)
})


test_that("DASNE analytical gradient is correct for single beta", {
  res <- embedder(dasne(beta = 0.5,
                        xi_eps = .Machine$double.eps, verbose = FALSE))
  fd_grad <- gradient_fd_xi(res, param_names = c("beta"))
  an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                 a2x(res$method$kernel$beta))
  expect_equal(an_grad, fd_grad, tol = 1e-6)
})

test_that("DSSNE analytical gradient is correct for single beta", {
  res <- embedder(dssne(beta = 0.5,
                        xi_eps = .Machine$double.eps, verbose = FALSE))
  fd_grad <- gradient_fd_xi(res, param_names = c("beta"))
  an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                 a2x(res$method$kernel$beta))
  expect_equal(an_grad, fd_grad, tol = 1e-6)
})



# Test inhomogeneous t-SNE ------------------------------------------------

# {h,i}t-SNE has no beta parameter so spared some complications
test_that("ht-SNE analytical gradient is correct for range of dof", {
  for (dof in c(1e-3, 0.01, 0.1, 1, 10, 100, 500)) {
    res <- embedder(htsne(dof = dof,
                          xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi(res, param_names = c("dof"))
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$dof))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(dof))
  }
})

test_that("it-SNE analytical gradient is correct for range of dof", {
  for (dof in c(1e-3, 0.01, 0.1, 1, 10, 100, 500)) {
    res <- embedder(itsne(dof = seq(dof, dof * 2, length.out = nr),
                          xi_eps = .Machine$double.eps, verbose = FALSE))

    fd_grad <- gradient_fd_xi_point(res, param_names = c("dof"))
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$dof))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(dof))
  }
})

test_that("it-3SNE analytical gradient is correct for range of dof", {
  for (dof in c(1e-3, 0.01, 0.1, 1, 10, 100, 500)) {
    res <- embedder(it3sne(dof = dof,
                          xi_eps = .Machine$double.eps, verbose = FALSE))
    fd_grad <- gradient_fd_xi_point(res, param_names = c("dof"))
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$dof))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(dof))
  }
})

test_that("it-SSNE analytical gradient is correct for range of dof", {
  for (dof in c(1e-3, 0.01, 0.1, 1, 10, 100, 500)) {
    res <- embedder(itssne(dof = seq(dof, dof * 2, length.out = nr),
                          xi_eps = .Machine$double.eps, verbose = FALSE))

    fd_grad <- gradient_fd_xi_point(res, param_names = c("dof"))
    an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                   a2x(res$method$kernel$dof))
    expect_equal(an_grad, fd_grad, tol = 1e-6, info = formatC(dof))
  }
})

test_that("it-SSNE works with generic parameter gradient too", {
  dof <- 0.1
  method <- lreplace(
    itssne(dof = seq(dof, dof * 2, length.out = nr),
           xi_eps = .Machine$double.eps, verbose = FALSE),
    gr_dof = itsne_cost_gr_param_plugin)

  res <- embedder(method)
  fd_grad <- gradient_fd_xi_point(res, param_names = c("dof"))
  an_grad <- res$method$extra_gr(res$opt, res$inp, res$out, res$method, 0,
                                 a2x(res$method$kernel$dof))
  expect_equal(an_grad, fd_grad, tol = 1e-6)
})


# Test Fixed Iteration Behavior -------------------------------------------

hssne_res <- quick_embed(method = hssne(alpha = 0.5, verbose = FALSE))
# HPSNE is HSSNE with cond P
# Q is joint by construction
hpsne_res <- quick_embed(method = lreplace(hssne_plugin(alpha = 0.5,
                                                        verbose = FALSE),
                                           prob_type = "cond"))
asne_res <- quick_embed(method = asne(beta = 0.5, verbose = FALSE))
tasne_res <- quick_embed(method = tasne(verbose = FALSE))

test_that("during fixed dof iterations ht-SNE behaves like tASNE if dof is fixed to 1", {
  res <- quick_embed(method = htsne(dof = 1, opt_iter = Inf,
                                    xi_eps = .Machine$double.eps,
                                    verbose = FALSE))
  expect_equal(res$report$costs, tasne_res$report$costs)
})

test_that("during fixed dof iterations ht-SNE behaves like ASNE with beta = 0.5 as dof -> Inf", {
  res <- quick_embed(method = htsne(dof = Inf, opt_iter = Inf,
                                    xi_eps = .Machine$double.eps,
                                    verbose = FALSE))
  expect_equal(res$report$costs, asne_res$report$costs)
})

test_that("during fixed dof iterations it-SNE behaves like tASNE if dof is fixed to 1", {
  res <- quick_embed(method = itsne(dof = 1, opt_iter = Inf,
                                    xi_eps = .Machine$double.eps,
                                    verbose = FALSE))
  expect_equal(res$report$costs, tasne_res$report$costs)
})

test_that("during fixed dof iterations it-SNE behaves like ASNE with beta = 0.5 as dof -> Inf", {
  res <- quick_embed(method = itsne(dof = Inf, opt_iter = Inf,
                                    xi_eps = .Machine$double.eps,
                                    verbose = FALSE))
  expect_equal(res$report$costs, asne_res$report$costs)
})

test_that("during fixed alpha iterations DHSSNE behaves like HSSNE", {
  res <- quick_embed(method = dhssne(alpha = 0.5, opt_iter = Inf,
                                     xi_eps = .Machine$double.eps,
                                     verbose = FALSE))
  expect_equal(res$report$costs, hssne_res$report$costs)
})

test_that("during fixed alpha iterations iHPSNE behaves like HPSNE", {
  res <- quick_embed(method = ihpsne(alpha = 0.5, opt_iter = Inf,
                                     xi_eps = .Machine$double.eps,
                                     verbose = FALSE))
  expect_equal(res$report$costs, hpsne_res$report$costs)
})


test_that("can dynamize HSSNE manually", {
  auto_dyn <- ihssne(alpha = 0.5, opt_iter = 0,
                     xi_eps = .Machine$double.eps, verbose = TRUE)
  auto_res <- quick_embed(method = auto_dyn)

  manual_dyn <- hssne(alpha = 0.5, verbose = FALSE)
  manual_dyn$opt_iter <- auto_dyn$opt_iter
  manual_dyn$dyn <- auto_dyn$dyn
  manual_dyn$switch_iter <- auto_dyn$switch_iter
  manual_dyn$xi_eps <- auto_dyn$xi_eps
  manual_dyn$alt_opt <- auto_dyn$alt_opt

  manual_dyn$dynamic_kernel <- TRUE

  manual_res <- quick_embed(method = manual_dyn)

  expect_equal(manual_res$report$costs, auto_res$report$costs, tol = 1e-4)
})

