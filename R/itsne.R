# Inhomogeneous t-SNE ----------------------------------------------------------

# inhomogeneous t-SNE
# Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S.
# (2016, October).
# t-Distributed Stochastic Neighbor Embedding with Inhomogeneous Degrees of
# Freedom.
# In International Conference on Neural Information Processing (pp. 119-128).
# Springer International Publishing.
# dof: 1 = tASNE, Inf = ASNE
# extra_gr = function for calculating derivative (must be KL divergence)
# itsne_cost_gr_param_plugin - always ok, requires cost gr
# (and hence plugin method)
# itsne_cost_gr_param - slightly more efficient, can't be used if forcing output
# probabilities to be joint (it-SSNE)
itsne <- function(dof = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  extra_gr = itsne_cost_gr_param,
                  eps = .Machine$double.eps, verbose = FALSE) {
  max_dof <- 1e8
  dof <- c(dof)
  dof[dof > max_dof] <- max_dof

  lreplace(
    asne_plugin(eps = eps, keep = c("qm", "wm", "d2m")),
    kernel = itsne_kernel(dof = dof),
    after_init_fn = function(inp, out, method) {
      nr <- nrow(out$ym)
      if (method$dyn == "point" && length(method$kernel$dof) != nr) {
        method$kernel$dof <- rep(method$kernel$dof, nr)
      }
      method$kernel <- check_symmetry(method$kernel)
      list(method = method)
    },
    opt_iter = opt_iter,
    dyn = "point",
    xi_eps = xi_eps,
    alt_opt = alt_opt,
    extra_gr = extra_gr,
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
    export_extra_par = function(method) {
      list(dof = method$kernel$dof)
    }
  )
}

# Homogeneous version of it-SNE with joint probs (i.e. "true" t-SNE)
# Single global dof parameter, sp can use more efficient gradient
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
  method <-
  lreplace(
    itsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          extra_gr = itsne_cost_gr_param_plugin,
          eps = eps, verbose = verbose),
    prob_type = 'joint'
  )
  method
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
    dyn = "global"
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
  if (iter < method$opt_iter) {
    return(rep(0, length(extra_par)))
  }
  xi <- extra_par
  dof <- xi * xi + method$xi_eps

  dw_ddof <- itsne_gr_param(out$d2m, out$wm, dof)
  gr_dof <- param_gr(method, inp, out, dw_ddof, method$dyn)

  2 * xi * gr_dof
}

# dC/d_xi
# This is a more efficient parameter gradient but won't work with enforced
# joint Q (i.e. can't be used with it-SSNE)
itsne_cost_gr_param <- function(opt, inp, out, method, iter, extra_par) {
  if (iter < method$opt_iter) {
    return(rep(0, length(extra_par)))
  }
  xi <- extra_par
  dof <- xi * xi + method$xi_eps

  c1 <- (out$d2m / dof) + 1
  hm <- log(c1) - ((out$d2m * (dof + 1)) / (dof * dof * c1))
  gr_dof <- hm * (inp$pm - out$qm)
  if (method$dyn == "point") {
    gr_dof <- rowSums(gr_dof)
  }
  else {
    gr_dof <- sum(gr_dof)
  }
  xi * gr_dof
}
