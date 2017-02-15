# A set of methods where the meta-parameters are also optimized.
# As these fluctuate over the course of the optimization, I call them
# "dynamic" SNE methods.

# Dynamic HSSNE -----------------------------------------------------------

# "Dynamic" HSSNE, alpha is optimized with coordinates
# Fully symmetric
dhssne <- function(beta = 1, alpha = 0, opt_iter = 0, xi_eps = 1e-3,
                   alt_opt = TRUE,
                   eps = .Machine$double.eps, verbose = FALSE) {
  ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
          eps = eps, verbose = verbose, alt_opt = alt_opt,
          dyn_beta = "static", dyn_alpha = "global")
}

dhasne <- function(beta = 1, alpha = 0, opt_iter = 0,
                   alt_opt = TRUE, xi_eps = 1e-3, eps = .Machine$double.eps,
                   verbose = FALSE) {
  lreplace(
    dhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, eps = eps,
           xi_eps = xi_eps, alt_opt = alt_opt, verbose = verbose),
    prob_type = "row")
}


# Dynamic Heavy-Tailed "Pair-wise" SNE
# Like DHSSNE but input probabilities are conditional. Output probabilities
# are joint by construction, due to global alpha, unless inhomogeneous betas
# are used.
# Mainly useful for comparison with iHPSNE
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
ihpsne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                   alt_opt = TRUE, xi_eps = 1e-3,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
            alt_opt = alt_opt, eps = eps,
            verbose = verbose, dyn_beta = "static", dyn_alpha = "point"),
    prob_type = "cond"
  )
}

# Inhomogeneous H3SNE: Semi Symmetric P is joint, Q is cond
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

ihasne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                   alt_opt = TRUE, xi_eps = 1e-3,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter,
            dyn_beta = "static", dyn_alpha = "point",
            alt_opt = alt_opt, xi_eps = xi_eps, eps = eps, verbose = verbose),
    prob_type = "row")
}


# Doubly Dynamic HSSNE ----------------------------------------------------

# dyn_ can be one of 'static', 'global', 'point'
ddhssne <- function(beta = 1, alpha = 0, xi_eps = 1e-3,
                    dyn_beta = "global", dyn_alpha = "global",
                    opt_iter = 0, switch_iter = opt_iter,
                    eps = .Machine$double.eps,
                    alt_opt = TRUE,
                    verbose = FALSE) {

  # If no inhomogeneity, then switch_iter is meaningless
  if (dyn_alpha == "global" && dyn_beta == "global") {
    switch_iter <- opt_iter
  }

  if (switch_iter < opt_iter) {
    stop("switch_iter must be >= opt_iter")
  }

  ok_dyn <- c("static", "global", "point")
  dyn_beta <- match.arg(dyn_beta, ok_dyn)
  dyn_alpha <- match.arg(dyn_alpha, ok_dyn)

  if (dyn_beta == "static" && dyn_alpha == "static") {
    stop("One of alpha and beta must be non-static")
  }

  alpha_fn <- sum
  if (dyn_alpha == "point") {
    alpha_fn <- rowSums
  }

  beta_fn <- sum
  if (dyn_beta == "point") {
    beta_fn <- rowSums
  }

  lreplace(
    hssne_plugin(beta = beta, alpha = alpha, eps = eps, verbose = verbose,
                 keep = c("qm", "wm", "d2m", "qcm")),
    after_init_fn = function(inp, out, method) {
      nr <- nrow(out$ym)
      if (dyn_alpha == "point" && length(alpha) != nr) {
        method$kernel$alpha <- rep(method$kernel$alpha, nr)
      }
      if (dyn_beta == "point" && length(beta) != nr) {
        method$kernel$beta <- rep(method$kernel$beta, nr)
      }
      method$kernel <- check_symmetry(method$kernel)
      list(method = method)
    },
    get_extra_par = function(method) {
      params <- c()
      if (dyn_alpha != "static") {
        params <- c(params, method$kernel$alpha)
      }
      if (dyn_beta != "static") {
        params <- c(params, method$kernel$beta)
      }

      xi <- params - method$xi_eps
      xi[xi < 0] <- 0
      sqrt(xi)
    },
    set_extra_par = function(method, extra_par) {
      xi <- extra_par
      params <- xi * xi + method$xi_eps
      if (dyn_alpha != "static") {
        method$kernel$alpha <- params[1:length(method$kernel$alpha)]
      }
      if (dyn_beta != "static") {
        if (dyn_alpha != "static") {
          beta_start <- length(method$kernel$alpha) + 1
        }
        else {
          beta_start <- 1
        }
        method$kernel$beta <- params[beta_start:length(params)]
      }
      method
    },
    extra_gr = function(opt, inp, out, method, iter, extra_par) {
      if (iter < method$opt_iter) {
        return(rep(0, length(extra_par)))
      }

      xi <- extra_par
      params <- xi * xi + method$xi_eps
      eps <- method$eps

      pm <- inp$pm
      fm <- out$d2m
      wm <- out$wm
      qm <- out$qm

      gr <- c()

      if (dyn_alpha != "static") {
        xi_alpha_range <- 1:(length(method$kernel$alpha))
        alpha <- params[xi_alpha_range]
        gr_alpha <- dhssne_gr_mat_alpha(pm, qm, fm, alpha, method$kernel$beta,
                                        out$qcm, eps)
        gr_alpha <- 2 * (xi[xi_alpha_range] / (alpha + eps)) * alpha_fn(gr_alpha)
        if (iter < switch_iter) {
          gr <- c(gr, rep(mean(gr_alpha), length(gr_alpha)))
        }
        else {
          gr <- c(gr, gr_alpha)
        }
      }

      if (dyn_beta != "static") {
        if (dyn_alpha != "static") {
          beta_start <- length(method$kernel$alpha) + 1
        }
        else {
          beta_start <- 1
        }
        xi_beta_range <- beta_start:length(xi)
        gr_beta <- dhssne_gr_mat_beta(pm, qm, fm, wm, method$kernel$alpha,
                                      out$qcm, eps)
        gr_beta <- 2 * xi[xi_beta_range] * beta_fn(gr_beta)

        if (iter < switch_iter) {
          gr <- c(gr, rep(mean(gr_beta), length(gr_beta)))
        }
        else {
          gr <- c(gr, gr_beta)
        }
      }

      gr
    },
    export_extra_par = function(method) {
      list(alpha = method$kernel$alpha,
           beta =  method$kernel$beta)
    },
    opt_iter = opt_iter,
    xi_eps = xi_eps,
    alt_opt = alt_opt
  )
}

dihssne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                   xi_eps = 1e-3, alt_opt = TRUE,
                   eps = .Machine$double.eps, verbose = FALSE) {
  ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, xi_eps = xi_eps,
          alt_opt = alt_opt, eps = eps,
          verbose = verbose, dyn_beta = "point", dyn_alpha = "point")
}

dihasne <- function(beta = 1, alpha = 0, opt_iter = 0, switch_iter = opt_iter,
                    xi_eps = 1e-3, alt_opt = TRUE,
                    eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    ddhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, eps = eps,
          verbose = verbose, dyn_beta = "point", dyn_alpha = "point",
          alt_opt = alt_opt, xi_eps = xi_eps),
    prob_type = "row")
}

# Gradients ---------------------------------------------------------------

# dw/da can be written as -h*w, which allows for simplification of dC/da
# This function returns the "-h" portion
gr_mult <- function(alpha, beta, fm, eps = .Machine$double.eps) {
  c1 <- alpha * beta * fm + 1
  (beta * fm) / (c1 + eps) - log(c1 + eps) / (alpha + eps)
}

# Gradient matrix dc/dalpha, used by both inhomogeneous HSSNE and dynamic HSSNE
dhssne_gr_mat_alpha <- function(pm, qm, fm, alpha, beta, qcm = NULL,
                                eps = .Machine$double.eps) {
  hm <- gr_mult(alpha, beta, fm, eps = eps)
  dhssne_gr_mat_kl(hm, pm, qm, qcm, eps)
}

dhssne_gr_mat_beta <- function(pm, qm, fm, wm, alpha, qcm = NULL,
                               eps = .Machine$double.eps) {
  hm <- fm * wm ^ alpha
  dhssne_gr_mat_kl(hm, pm, qm, qcm, eps)
}

dhssne_gr_mat_kl <- function(hm, pm, qm, qcm = NULL,
                             eps = .Machine$double.eps) {
  if (!is.null(qcm)) {
    # Presence of qcm indicates an asymmetric kernel with a joint
    # probability: can't simplify expression as much
    gr <- hm * qcm * (pm / (qm + eps) - 1)
  }
  else {
    # pairwise and semi-symmetric versions can use simpler version
    gr <- hm * (pm - qm)
  }
  gr
}
