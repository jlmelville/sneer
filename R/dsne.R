# A set of methods where the kernel parameters are also optimized.
# As these fluctuate over the course of the optimization, I call them
# "dynamic" SNE methods.

# See also the inhomogeneous (it-SNE) functions

# Dynamic HSSNE -----------------------------------------------------------

# "Dynamic" HSSNE, alpha is optimized with coordinates
# Fully symmetric
# Can use DDHSSNE default symm KL param gradient iff beta is uniform. Would also
# need to do this if we had a static-alpha/dynamic-beta method and static alphas
# can be non-uniform (we don't currently do either, though)
dhssne <- function(beta = 1, alpha = 0, opt_iter = 0, xi_eps = 1e-3,
                   alt_opt = TRUE,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
            eps = eps, verbose = verbose, alt_opt = alt_opt),
    dyn = list(alpha = "global", beta = "static")
  )
}

# Asymmetric version of DHSSNE
# Can use DDHSSNE default symm KL param gradient
dhasne <- function(beta = 1, alpha = 0, opt_iter = 0,
                   alt_opt = TRUE, xi_eps = 1e-3, eps = .Machine$double.eps,
                   verbose = FALSE) {
  lreplace(
    dhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, eps = eps,
           xi_eps = xi_eps, alt_opt = alt_opt, verbose = verbose),
    prob_type = "row"
  )
}


# Dynamic Heavy-Tailed "Pair-wise" SNE
# Like DHSSNE but input probabilities are conditional. Output probabilities
# are joint by construction, due to global alpha, unless inhomogeneous betas
# are used.
# Mainly useful for comparison with iHPSNE
# Can use DDHSSNE default symm KL param gradient
dhpsne <- function(beta = 1, alpha = 0, opt_iter = 0, alt_opt = TRUE,
                   xi_eps = 1e-3,
                   eps = .Machine$double.eps,
                   verbose = FALSE) {
  lreplace(
    dhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
           alt_opt = alt_opt, eps = eps,
           verbose = verbose),
    prob_type = "cond"
  )
}

# Dynamic Heavy-Tailed Semi Symmetric SNE
# P is joint, Q is cond
# Can use DDHSSNE default symm KL param gradient
dh3sne <- function(beta = 1, alpha = 0, opt_iter = 0, alt_opt = TRUE,
                   xi_eps = 1e-3,
                   eps = .Machine$double.eps,
                   verbose = FALSE) {
  lreplace(
    dhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
           alt_opt = alt_opt, eps = eps,
           verbose = verbose),
    out_prob_type = "cond"
  )
}


# Inhomogeneous HSSNE -----------------------------------------------------
# Point wise values of alpha

# inhomogeneous Heavy-Tailed "Pair-wise" SNE, P and Q are cond
# symm KL param gradient
ihpsne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                   alt_opt = TRUE, xi_eps = 1e-3,
                   gr_alpha = NULL,
                   gr_beta = NULL,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
            alt_opt = alt_opt, eps = eps, verbose = verbose),
    prob_type = "cond",
    dyn = list(alpha = "point", beta = "static")
  )
}

# Inhomogeneous H3SNE: Semi Symmetric P is joint, Q is cond
# Can use DDHSSNE default symm KL param gradient
ih3sne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                   alt_opt = TRUE, xi_eps = 1e-3,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ihpsne(beta = beta, alpha = alpha, opt_iter = opt_iter,
           switch_iter = switch_iter, xi_eps = xi_eps,
           alt_opt = alt_opt, eps = eps, verbose = verbose),
    prob_type = "joint",
    out_prob_type = "cond"
  )
}

# Inhomogeneous HSSNE: P is joint, Q is joint (by averaging)
# Can use DDHSSNE default asymm KL param gradient (NOT the symmetric version)
ihssne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                   alt_opt = TRUE, xi_eps = 1e-3,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ihpsne(beta = beta, alpha = alpha, opt_iter = opt_iter,
           switch_iter = switch_iter, xi_eps = xi_eps,
           alt_opt = alt_opt, eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# Inhomogeneous HASNE: the heavy-tailed equivalent of it-SNE
# Can use DDHSSNE default symm KL param gradient
ihasne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                   alt_opt = TRUE, xi_eps = 1e-3,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter,
            alt_opt = alt_opt, xi_eps = xi_eps, eps = eps, verbose = verbose),
    prob_type = "row",
    dyn = list(alpha = "point", beta = "static")
  )
}


# Doubly Dynamic HSSNE ----------------------------------------------------

# dyn - list mapping from parameter name (alpha and beta) to one of: 'static'
# (don't optimize), 'global' (one value), 'point' (allow each parameter to vary
# per point)
# If you are transferring precisions, it makes no sense to specify a 'global'
# dynamic beta. So don't do that.
# opt_iter - at what iteration to start optimizing the parameters
# switch_iter - if doing 'point' optimization, treat parameters as global
# between opt_iter and switch_iter
# gr_alpha - function to use for calculating the alpha gradient. If NULL, we
#  check to see if we are using an asymmetric kernel with joint output
#  probabilities. If so, we use the safe, but slow plugin gradient. Otherwise,
#  we can use a faster version, but this assumes that we are using the KL
#  divergence as a cost function (a likely scenario). If not, provide your
#  own function here.
# gr_beta - function to use for calculating the beta gradient. Same rules apply
#  here as with gr_alpha.
# NB an asymmetric kernel results from using a non-uniform alpha or beta, even
# if you aren't optimizing them, e.g. transferring precisions.
# Because we assume a uniform beta, we can use the symm KL parameter gradient
ddhssne <- function(beta = 1, alpha = 0, xi_eps = 1e-3,
                    opt_iter = 0, switch_iter = opt_iter,
                    eps = .Machine$double.eps,
                    alt_opt = TRUE,
                    verbose = FALSE) {

  if (switch_iter < opt_iter) {
    stop("switch_iter must be >= opt_iter")
  }

  lreplace(
    hssne_plugin(beta = beta, alpha = alpha, eps = eps, verbose = verbose),
    dynamic_kernel = TRUE,
    dyn = list(alpha = "global", beta = "global"),
    opt_iter = opt_iter,
    switch_iter = switch_iter,
    xi_eps = xi_eps,
    alt_opt = alt_opt
  )
}

# Use Asymmetric KL kernel parameter gradient
dihssne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                    xi_eps = 1e-3, alt_opt = TRUE,
                    eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
            alt_opt = alt_opt, eps = eps, verbose = verbose),
    dyn = list(alpha = "point", beta = "point")
  )
}

# Use Symmetric KL kernel parameter gradient
dihasne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                    xi_eps = 1e-3, alt_opt = TRUE,
                    eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    dihssne(beta = beta, alpha = alpha, opt_iter = opt_iter, eps = eps,
            verbose = verbose,
            alt_opt = alt_opt, xi_eps = xi_eps),
    prob_type = "row"
  )
}

# Dynamic ASNE ------------------------------------------------------------

# Optimizes beta parameter of exponential kernel. Less complex due to only
# one parameter being involved, but probably not that useful.
# gr_beta is the parameter gradient fn - can leave NULL and let after_init_fn
# decide the best choice
iasne <- function(beta = 1, eps = .Machine$double.eps, verbose = FALSE,
                  opt_iter = 0, xi_eps = 1e-3, alt_opt = TRUE) {
  lreplace(
    asne_plugin(beta = beta, eps = eps, verbose = verbose),
    dynamic_kernel = TRUE,
    dyn = list(beta = "point"),
    opt_iter = opt_iter,
    xi_eps = xi_eps,
    alt_opt = alt_opt
  )
}

# Don't use this with non-uniform beta
dasne <- function(beta = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    iasne(beta = beta, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    dyn = list(beta = "global")
  )
}

issne <- function(beta = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    iasne(beta = beta, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# Don't use this with non-uniform beta
dssne <- function(beta = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    issne(beta = beta, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    dyn = list(beta = "global")
  )
}

# Gradients ---------------------------------------------------------------

# Generic functions which can be used with any cost, kernel and normalization
# gradient for kernel parameters

# Calculates dC/d_param
param_gr <- function(method, inp, out, dw_dparam, dyn) {
  gr <- dw_dparam * param_stiffness_norm(method, inp, out)
  if (dyn == "point") {
    rowSums(gr)
  }
  else {
    sum(gr)
  }
}

# Calcultes dC/dw accounting for the type of normalization (if any)
param_stiffness_norm <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)
  if (!is.null(method$out_prob_type) && method$out_prob_type == "un") {
    norm_term <- dc_dq
  }
  else if (method$prob_type == "row") {
    wm_sum <- rowSums(out$wm)
    norm_term <- rowSums(dc_dq * out$qm)
    norm_term <- sweep(dc_dq, 1, norm_term) # subtract row sum from each row element
    norm_term <- norm_term / wm_sum
  }
  else {
    wm_sum <- sum(out$wm)
    norm_term <- (dc_dq - sum(dc_dq * out$qm)) / wm_sum
  }
  norm_term
}

# Heavy Tail Parameter Gradient -------------------------------------------

# These functions are used only with DHSSNE and related methods so don't
# need to be with the generic kernels

# dw/d_alpha
heavy_tail_gr_alpha <- function(d2m, beta = 1, alpha = 1.5e-8,
                                wm = heavy_tail_weight(d2m, beta = beta,
                                                       alpha = alpha),
                                eps = .Machine$double.eps) {
  c1 <- alpha * beta * d2m + 1
  c1e <- c1 + eps
  hm <- log(c1e) / (alpha * alpha + eps) - (beta * d2m) / (alpha * c1e)
  wm * hm
}

# dw/d_beta
heavy_tail_gr_beta <- function(d2m, beta = 1, alpha = 1.5e-8,
                               wm = heavy_tail_weight(d2m, beta = beta,
                                                      alpha = alpha)) {
  hm <- -d2m * wm ^ alpha
  wm * hm
}

# A generic dC/d_xi(alpha)
heavy_tail_cost_gr_alpha_plugin <- function(inp, out, method, xi) {
  xi_alpha_range <- 1:(length(method$kernel$alpha))

  dw_da <- heavy_tail_gr_alpha(out$d2m, method$kernel$beta, method$kernel$alpha,
                               out$wm, method$eps)
  gr_alpha <- param_gr(method, inp, out, dw_da, method$dyn$alpha)
  2 * xi[xi_alpha_range] * gr_alpha
}

# A generic dC/d_xi(beta)
heavy_tail_cost_gr_beta_plugin <- function(inp, out, method, xi) {
  if (method$dyn$alpha != "static") {
    beta_start <- length(method$kernel$alpha) + 1
  }
  else {
    beta_start <- 1
  }
  xi_beta_range <- beta_start:length(xi)

  dw_db <- heavy_tail_gr_beta(out$d2m, method$kernel$beta, method$kernel$alpha,
                              out$wm)
  gr_beta <- param_gr(method, inp, out, dw_db, method$dyn$beta)
  2 * xi[xi_beta_range] * gr_beta
}

# A more efficient dC/d_xi(alpha) if using the KL divergence as a cost function
# and a symmetric kernel
heavy_tail_cost_gr_alpha_kl_symm <- function(inp, out, method, xi) {
  xi_alpha_range <- 1:(length(method$kernel$alpha))

  alpha <- method$kernel$alpha + method$eps
  beta <- method$kernel$beta
  c1 <- alpha * beta * out$d2m + 1 + method$eps
  gr <- ((beta * out$d2m) / c1 - log(c1) / alpha) / alpha
  gr <- gr * (inp$pm - out$qm)
  if (method$dyn$alpha == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  2 * xi[xi_alpha_range] * gr
}

# A more efficient dC/d_xi(beta) if using the KL divergence as a cost function
# and a symmetric kernel
heavy_tail_cost_gr_beta_kl_symm <- function(inp, out, method, xi) {
  if (method$dyn$alpha != "static") {
    beta_start <- length(method$kernel$alpha) + 1
  }
  else {
    beta_start <- 1
  }
  xi_beta_range <- beta_start:length(xi)

  gr <- out$d2m * out$wm ^ method$kernel$alpha
  gr <- gr * (inp$pm - out$qm)
  if (method$dyn$beta == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  2 * xi[xi_beta_range] * gr
}

# A more efficient dC/d_xi(alpha) if using the KL divergence as a cost function
# and an asymmetric kernel
heavy_tail_cost_gr_alpha_kl_asymm <- function(inp, out, method, xi) {
  xi_alpha_range <- 1:(length(method$kernel$alpha))

  alpha <- method$kernel$alpha + method$eps
  beta <- method$kernel$beta
  c1 <- alpha * beta * out$d2m + 1 + method$eps
  gr <- ((beta * out$d2m) / c1 - log(c1) / alpha) / alpha
  gr <- gr * out$qcm * ((inp$pm / (out$qm + method$eps)) - 1)
  if (method$dyn$alpha == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  2 * xi[xi_alpha_range] * gr
}

# A more efficient dC/d_xi(beta) if using the KL divergence as a cost function
# and an asymmetric kernel
heavy_tail_cost_gr_beta_kl_asymm <- function(inp, out, method, xi) {
  if (method$dyn$alpha != "static") {
    beta_start <- length(method$kernel$alpha) + 1
  }
  else {
    beta_start <- 1
  }
  xi_beta_range <- beta_start:length(xi)

  gr <- out$d2m * out$wm ^ method$kernel$alpha
  gr <- gr * out$qcm * ((inp$pm / (out$qm + method$eps)) - 1)
  if (method$dyn$beta == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  2 * xi[xi_beta_range] * gr
}

# dC/d_xi
heavy_tail_gr_param_xi <- function(kernel, inp, out, method, xi, iter) {
  gr_alpha <- c()
  if (method$dyn$alpha != "static") {
    gr_alpha <- method$gr_alpha(inp, out, method, xi)
    if (iter < method$switch_iter) {
      gr_alpha <- rep(mean(gr_alpha), length(gr_alpha))
    }
  }

  gr_beta <- c()
  if (method$dyn$beta != "static") {
    gr_beta <- method$gr_beta(inp, out, method, xi)
    if (iter < method$switch_iter) {
      gr_beta <- rep(mean(gr_beta), length(gr_beta))
    }
  }

  c(gr_alpha, gr_beta)
}

heavy_tail_params <- function(kernel) {
  list(alpha = kernel$alpha, beta =  kernel$beta)
}

heavy_tail_params_xi <- function(kernel, method) {
  params <- c()
  if (method$dyn$alpha != "static") {
    params <- c(params, kernel$alpha)
  }
  if (method$dyn$beta != "static") {
    params <- c(params, kernel$beta)
  }

  xi <- params - method$xi_eps
  xi[xi < 0] <- 0
  sqrt(xi)
}

set_heavy_tail_params_xi <- function(kernel, method, xi) {
  params <- xi * xi + method$xi_eps
  if (method$dyn$alpha != "static") {
    kernel$alpha <- params[1:length(kernel$alpha)]
  }
  if (method$dyn$beta != "static") {
    if (method$dyn$alpha != "static") {
      beta_start <- length(kernel$alpha) + 1
    }
    else {
      beta_start <- 1
    }
    kernel$beta <- params[beta_start:length(params)]
  }
  kernel
}


# Exponential Parameter Gradient ------------------------------------------

# dC/d_xi generic for exponential kernel
exp_cost_gr_param_plugin <- function(opt, inp, out, method, iter, extra_par) {
  xi <- extra_par

  dw_dbeta <- exp_gr_param(out)
  gr <- param_gr(method, inp, out, dw_dbeta, method$dyn$beta)

  2 * xi * gr
}

# dc/d_xi for exponential kernel and KL divergence
# Can only be used with joint output probabilities if the kernel is symmetric
# (i.e. uniform beta)
exp_cost_gr_param_kl_symm <- function(opt, inp, out, method, iter, extra_par) {
  xi <- extra_par

  gr <- out$d2m * (inp$pm - out$qm)
  if (method$dyn$beta == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  2 * xi * gr
}

# dc/d_xi for exponential kernel and KL divergence
# Can be used with asymmetric or symmetric kernel and any output probability
# type - but exp_cost_gr_param_kl_symm is a better choice for non-joint output
# probabilities
exp_cost_gr_param_kl_asymm <- function(opt, inp, out, method, iter, extra_par) {
  xi <- extra_par

  gr <- out$d2m * out$qcm * ((inp$pm / (out$qm + method$eps)) - 1)
  if (method$dyn$beta == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  2 * xi * gr
}

# dw/d_beta
exp_gr_param <- function(out) {
  -out$d2m * out$wm
}


# Dynamize Kernels ---------------------------------------------------------

# should be called by make_kernel_dynamic, delegating to kernel$make_dynamic
# as part of before_init

dynamize_exp_kernel <- function(method) {
  method$dyn$after_init_fn <- function(inp, out, method) {
    nr <- nrow(out$ym)
    if (method$dyn$beta == "point" && length(method$kernel$beta) != nr) {
      method$kernel$beta <- rep(method$kernel$beta, nr)
    }
    method$kernel <- check_symmetry(method$kernel)

    # Leaving method null means we are dynamizing a method manually
    if (is.null(method$gr_beta)) {
      if (method$cost$name == "KL") {
        if (is_joint_out_prob(method) && is_asymmetric_kernel(method$kernel)) {
          if (method$verbose) {
            message("Using KL cost + asymmetric kernel parameter gradients")
          }
          method$gr_beta <- exp_cost_gr_param_kl_asymm
          method$out_keep <- unique(c(method$out_keep, "qcm", "d2m"))
        }
        else {
          if (method$verbose) {
            message("Using KL cost + symmetric kernel parameter gradients")
          }
          method$gr_beta <- exp_cost_gr_param_kl_symm
          method$out_keep <- unique(c(method$out_keep, "d2m"))
        }
      }
      else {
        # But we can fall back to the plugin gradient which is always safe
        if (method$verbose) {
          message("Using plugin parameter gradients")
        }
        method$gr_beta <- exp_cost_gr_param_plugin
        method$out_keep <- unique(c(method$out_keep, "d2m", "wm"))
      }
    }

    list(method = method)
  }


  lreplace(method,
  extra_gr = function(opt, inp, out, method, iter, extra_par) {
    if (iter < method$opt_iter) {
      return(rep(0, length(extra_par)))
    }

    method$gr_beta(opt, inp, out, method, iter, extra_par)
  },
  get_extra_par = function(method) {
    xi <- method$kernel$beta - method$xi_eps
    xi[xi < 0] <- xi
    sqrt(xi)
  },
  set_extra_par = function(method, extra_par) {
    xi <- extra_par
    method$kernel$beta <- xi * xi + method$xi_eps
    method
  },
  export_extra_par = function(method) {
    list(beta = method$kernel$beta)
  }
  )
}

dynamize_heavy_tail_kernel <- function(method) {
  method$dyn$after_init_fn <- function(inp, out, method) {
    nr <- nrow(out$ym)
    kernel <- method$kernel
    if (method$dyn$alpha == "point" && length(kernel$alpha) != nr) {
      kernel$alpha <- rep(kernel$alpha, nr)
    }
    if (method$dyn$beta == "point" && length(kernel$beta) != nr) {
      kernel$beta <- rep(kernel$beta, nr)
    }
    method$kernel <- check_symmetry(kernel)

    # Leaving method null means we are dynamizing a method manually OR we are
    # using the KL cost and would like to use a slightly more efficient gradient
    # calculation, but can't be sure if a static parameter is uniform or not
    if (is.null(method$gr_alpha) || is.null(method$gr_beta)) {
      if (method$cost$name == "KL") {
        if (is_joint_out_prob(method) && is_asymmetric_kernel(method$kernel)) {
          if (method$verbose) {
            message("Using KL cost + asymmetric kernel parameter gradients")
          }
          method$gr_alpha <- heavy_tail_cost_gr_alpha_kl_asymm
          method$gr_beta <- heavy_tail_cost_gr_beta_kl_asymm
          method$out_keep <- unique(c(method$out_keep, "qcm", "d2m"))
        }
        else {
          if (method$verbose) {
            message("Using KL cost + symmetric kernel parameter gradients")
          }
          method$gr_alpha <- heavy_tail_cost_gr_alpha_kl_symm
          method$gr_beta <- heavy_tail_cost_gr_beta_kl_symm
          method$out_keep <- unique(c(method$out_keep, "d2m"))
        }
      }
      else {
        # But we can fall back to the plugin gradient which is always safe
        if (method$verbose) {
          message("Using plugin parameter gradients")
        }
        method$gr_alpha <- heavy_tail_cost_gr_alpha_plugin
        method$gr_beta <- heavy_tail_cost_gr_beta_plugin
        method$out_keep <- unique(c(method$out_keep, "d2m", "wm"))
      }
    }

    list(method = method)
  }


  lreplace(method,
  get_extra_par = function(method) {
    heavy_tail_params_xi(method$kernel, method)
  },
  set_extra_par = function(method, extra_par) {
    method$kernel <- set_heavy_tail_params_xi(method$kernel, method,
                                              extra_par)
    method
  },
  extra_gr = function(opt, inp, out, method, iter, extra_par) {
    if (iter < method$opt_iter) {
      return(rep(0, length(extra_par)))
    }

    heavy_tail_gr_param_xi(method$kernel, inp, out, method, xi = extra_par,
                           iter)

  },
  export_extra_par = function(method) {
    heavy_tail_params(method$kernel)
  }
  )
}
