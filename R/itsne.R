# Inhomogeneous t-SNE ----------------------------------------------------------

# homogeneous t-SNE, single global dof parameter
# dof: 1 = tASNE, Inf = ASNE
htsne <- function(dof = 1, eps = .Machine$double.eps, opt_iter = 0,
                  verbose = FALSE) {
  dof <- base::min(dof, 1e8)
  lreplace(
    asne_plugin(eps = eps, keep = c("qm", "wm", "d2m")),
    kernel = itsne_kernel(dof = dof),
    get_extra_par = function(method) {
      sqrt(method$kernel$dof - method$eps)
    },
    set_extra_par = function(method, extra_par) {
      xi <- extra_par
      method$kernel$dof <- xi * xi + method$eps
      method
    },
    extra_gr = function(opt, inp, out, method, iter, extra_par) {
      if (iter < method$opt_iter) {
        return(rep(0, length(extra_par)))
      }
      eps <- method$eps
      xi <- extra_par
      dof <- xi * xi + method$eps
      fm <- out$d2m
      c1 <- (fm / dof) + 1

      hm <- log(c1) - ((fm * (dof + 1)) / (dof * dof * c1))
      gr <- hm * (inp$pm - out$qm)

      xi * sum(gr)
    },
    export_extra_par = function(method) {
      list(dof = method$kernel$dof)
    },
    opt_iter = opt_iter
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
                  eps = .Machine$double.eps, verbose = FALSE) {
  lreplace(
    htsne(dof = dof, opt_iter = opt_iter, eps = eps, verbose = verbose),
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
      eps <- method$eps
      xi <- extra_par
      dof <- xi * xi + method$eps
      fm <- out$d2m
      c1 <- (fm / dof) + 1

      hm <- log(c1) - ((fm * (dof + 1)) / (dof * dof * c1))
      gr <- hm * (inp$pm - out$qm)

      xi * rowSums(gr)
    }
  )
}
