# Inhomogeneous t-SNE ----------------------------------------------------------

# inhomogeneous t-SNE
# Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S.
# (2016, October).
# t-Distributed Stochastic Neighbor Embedding with Inhomogeneous Degrees of
# Freedom.
# In International Conference on Neural Information Processing (pp. 119-128).
# Springer International Publishing.
# dof: 1 = tASNE, Inf = ASNE
# param fn is gr_dof, can leave NULL and let after_init_fn work it out
itsne <- function(dof = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  max_dof <- 1e8
  dof <- c(dof)
  dof[dof > max_dof] <- max_dof

  lreplace(
    asne(eps = eps),
    kernel = itsne_kernel(dof = dof),
    dynamic_kernel = TRUE,
    dyn = list(dof = "point"),
    opt_iter = opt_iter,
    xi_eps = xi_eps,
    alt_opt = alt_opt
  )
}

# Homogeneous version of it-SNE with joint probs (i.e. "true" t-SNE)
# Single global dof parameter, so can use more efficient gradient
# dof: 1 = tASNE, Inf = ASNE
htssne <- function(dof = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    htsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# inhomogeneous version of t-SNE: with asymmetric kernel dof, must enforce
# joint Q by averaging. Can't use simplified gradient expression
itssne <- function(dof = 1, opt_iter = 0,
                   xi_eps = 1e-3, alt_opt = TRUE,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    itsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    prob_type = 'joint'
  )
}

# P is joint, Q is conditional, i.e. don't bother averaging an complicating
# gradient
it3sne <- function(dof = 1, opt_iter = 0,
                   xi_eps = 1e-3, alt_opt = TRUE,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    itsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    out_prob_type = 'cond',
    prob_type = 'joint'
  )
}

# homogeneous t-SNE, single global dof parameter
# dof: 1 = tASNE, Inf = ASNE
htsne <- function(dof = 1, opt_iter = 0, xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    itsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    dyn = list(dof = "global")
  )
}

# Gradient ----------------------------------------------------------------

# dw/d_dof
itsne_gr_param <- function(fm, wm, dof) {
  c1 <- (fm / dof) + 1
  hm <- ((fm * (dof + 1)) / (dof * dof * c1)) - log(c1)
  0.5 * hm * wm
}

# dC/d_xi
# This is a generic parameter gradient: works with enforced joint Q
itsne_cost_gr_param_plugin <- function(opt, inp, out, method, iter, extra_par) {
  xi <- extra_par
  dof <- xi * xi + method$xi_eps

  dw_ddof <- itsne_gr_param(out$d2m, out$wm, dof)
  gr <- param_gr(method, inp, out, dw_ddof, method$dyn$dof)

  2 * xi * gr
}

# dC/d_xi
# This is a more efficient parameter gradient but won't work with enforced
# joint Q (i.e. can't be used with it-SSNE)
itsne_cost_gr_param <- function(opt, inp, out, method, iter, extra_par) {
  xi <- extra_par
  dof <- xi * xi + method$xi_eps

  c1 <- (out$d2m / dof) + 1
  hm <- log(c1) - ((out$d2m * (dof + 1)) / (dof * dof * c1))
  gr <- hm * (inp$pm - out$qm)
  if (method$dyn$dof == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  xi * gr
}

# For KL cost with asymmetric kernel and joint output probability
itsne_cost_gr_param_asymm <- function(opt, inp, out, method, iter, extra_par) {
  xi <- extra_par
  dof <- xi * xi + method$xi_eps

  c1 <- (out$d2m / dof) + 1
  gr <- log(c1) - ((out$d2m * (dof + 1)) / (dof * dof * c1))
  gr <- gr * out$qcm * ((inp$pm / (out$qm + method$eps)) - 1)
  if (method$dyn$dof == "point") {
    gr <- rowSums(gr)
  }
  else {
    gr <- sum(gr)
  }
  xi * gr
}

# Dynamize Kernel ---------------------------------------------------------

# should be called by make_kernel_dynamic, delegating to kernel$make_dynamic
# as part of before_init
dynamize_inhomogeneous_kernel <- function(method) {
  if (is.null(method$dyn$dof)) {
    method$dyn$dof <- "global"
  }
  if (method$dyn$dof == "static") {
    if (method$verbose) {
      message("Kernel parameters are all marked as 'static', no optimization")
    }
    method
  }
  else {
    method$dyn$after_init_fn <- function(inp, out, method) {
      nr <- nrow(out$ym)
      if (method$dyn$dof == "point" && length(method$kernel$dof) != nr) {
        method$kernel$dof <- rep(method$kernel$dof, nr)
      }
      if (method$dyn$dof == "point") {
        method$kernel <- set_kernel_asymmetric(method$kernel)
      }
      else {
        method$kernel <- check_symmetry(method$kernel)
      }

      # Leaving method null means we are dynamizing a method manually
      if (is.null(method$gr_dof)) {
        if (method$cost$name == "KL") {
          if (is_joint_out_prob(method) && is_asymmetric_kernel(method$kernel)) {
            if (method$verbose) {
              message("Using KL cost + asymmetric kernel parameter gradients")
            }
            method$gr_dof <- itsne_cost_gr_param_asymm
            method$out_keep <- unique(c(method$out_keep, "qcm", "d2m"))
          }
          else {
            if (method$verbose) {
              message("Using KL cost + symmetric kernel parameter gradients")
            }
            method$gr_dof <- itsne_cost_gr_param
            method$out_keep <- unique(c(method$out_keep, "d2m"))
          }
        }
        else {
          # But we can fall back to the plugin gradient which is always safe
          if (method$verbose) {
            message("Using plugin parameter gradients")
          }
          method$gr_dof <- itsne_cost_gr_param_plugin
          method$out_keep <- unique(c(method$out_keep, "wm", "d2m"))
        }
      }

      list(method = method)
    }

    lreplace(
      method,
      get_extra_par = function(method) {
        xi <- method$kernel$dof - method$xi_eps
        xi[xi < 0] <- xi
        sqrt(xi)
      },
      set_extra_par = function(method, extra_par) {
        xi <- extra_par
        method$kernel$dof <- xi * xi + method$xi_eps
        method
      },
      extra_gr = function(opt, inp, out, method, iter, extra_par) {
        if (iter < method$opt_iter) {
          return(rep(0, length(extra_par)))
        }

        method$gr_dof(opt, inp, out, method, iter, extra_par)
      },
      export_extra_par = function(method) {
        list(dof = method$kernel$dof)
      })
  }
}
