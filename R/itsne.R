# Inhomogeneous t-SNE ----------------------------------------------------------

# homogeneous t-SNE, single global dof parameter
# dof: 1 = tASNE, Inf = ASNE
htsne <- function(dof = 1, opt_iter = 0, xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  dof <- base::min(dof, 1e8)
  lreplace(
    asne_plugin(eps = eps, keep = c("qm", "wm", "d2m")),
    kernel = itsne_kernel(dof = dof),
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
      xi <- extra_par
      dof <- xi * xi + method$xi_eps

      xi * sum(itsne_gr_mat(inp$pm, out$d2m, out$qm, dof))
    },
    export_extra_par = function(method) {
      list(dof = method$kernel$dof)
    },
    opt_iter = opt_iter,
    xi_eps = xi_eps,
    alt_opt = alt_opt
  )
}


htssne <- function(dof = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    htsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# inhomogeneous t-SNE
# Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S.
# (2016, October).
# t-Distributed Stochastic Neighbor Embedding with Inhomogeneous Degrees of
# Freedom.
# In International Conference on Neural Information Processing (pp. 119-128).
# Springer International Publishing.
# dof: 1 = tASNE, Inf = ASNE
itsne <- function(dof = 1, opt_iter = 0,
                  xi_eps = 1e-3, alt_opt = TRUE,
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    htsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    after_init_fn = function(inp, out, method) {
      nr <- nrow(out$ym)
      dof <- method$kernel$dof
      if (length(dof) != nr) {
        method$kernel$dof <- rep(dof, nr)
      }
      method$kernel <- check_symmetry(method$kernel)
      list(method = method)
    },
    extra_gr = function(opt, inp, out, method, iter, extra_par) {
      if (iter < method$opt_iter) {
        return(rep(0, length(extra_par)))
      }
      xi <- extra_par
      dof <- xi * xi + method$xi_eps

      xi * rowSums(itsne_gr_mat(inp$pm, out$d2m, out$qm, dof))
    }
  )
}

itssne <- function(dof = 1, opt_iter = 0,
                   xi_eps = 1e-3, alt_opt = TRUE,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    itsne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    prob_type = 'joint'
  )
}

it3sne <- function(dof = 1, opt_iter = 0,
                   xi_eps = 1e-3, alt_opt = TRUE,
                   eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    itssne(dof = dof, opt_iter = opt_iter, xi_eps = xi_eps, alt_opt = alt_opt,
          eps = eps, verbose = verbose),
    out_prob_type = 'cond'
  )
}

itsne_gr_mat <- function(pm, fm, qm, dof) {
  c1 <- (fm / dof) + 1
  hm <- log(c1) - ((fm * (dof + 1)) / (dof * dof * c1))
  hm * (pm - qm)
}
