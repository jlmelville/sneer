# A set of methods where the meta-parameters are also optimized.
# As these fluctuate over the course of the optimization, I call them
# "dynamic" SNE methods.

# "Dynamic" HSSNE, alpha is optimized with coordinates
dhssne <- function(beta = 1, alpha = 0, opt_iter = 0, eps = .Machine$double.eps,
                   verbose = TRUE) {
  lreplace(
    hssne_plugin(beta = beta, alpha = alpha, eps = eps, verbose = verbose),
    update_out_fn = make_update_out(keep = c("qm", "wm", "d2m")),
    get_extra_par = function(method) {
      alpha <- method$kernel$alpha
      sqrt(alpha - method$eps)
    },
    set_extra_par = function(method, extra_par) {
      xi <- extra_par
      method$kernel$alpha <- xi * xi + method$eps
      method
    },
    extra_gr = function(opt, inp, out, method, iter, extra_par) {
      if (iter < method$opt_iter) {
        return(rep(0, length(extra_par)))
      }
      xi <- extra_par
      alpha <- xi * xi + method$eps
      fm <- out$d2m
      c1 <- alpha * fm + 1
      hm <- (fm / (alpha * c1)) - (log(c1) / (alpha * alpha))
      gr <- hm * (inp$pm - out$qm)
      2 * xi * sum(gr)
    },
    export_extra_par = function(method) {
      list(alpha = method$kernel$alpha)
    },
    opt_iter = opt_iter
  )
}

# Inhomogeneous HSSNE: alpha is defined per point
ihssne <- function(beta = 1, alpha = 0, opt_iter = 0, eps = .Machine$double.eps,
                   verbose = TRUE) {
  lreplace(
    dhssne(beta = beta, alpha = alpha, opt_iter = opt_iter, eps = eps,
           verbose = verbose),
    after_init_fn = function(inp, out, method) {
      nr <- nrow(out$ym)
      alpha <- method$kernel$alpha
      if (length(alpha) != nr) {
        method$kernel$alpha <- rep(alpha, nr)
      }
      method$kernel <- check_symmetry(method$kernel)
      list(method = method)
    },
    set_extra_par = function(method, extra_par) {
      xi <- extra_par
      method$kernel$alpha <- xi * xi + method$eps
      method$kernel <- check_symmetry(method$kernel)
      method
    },
    extra_gr = function(opt, inp, out, method, iter, extra_par) {
      if (iter < method$opt_iter) {
        return(rep(0, length(extra_par)))
      }
      xi <- extra_par
      alpha <- xi * xi + method$eps
      fm <- out$d2m
      c1 <- alpha * fm + 1

      hm <- (fm / c1) - (log(c1) / alpha)
      gr <- hm * (inp$pm - out$qm)
      2 * (xi / alpha) * rowSums(gr)
    },
    prob_type = "cond"
  )
}

# homogeneous t-SNE, single global dof parameter
# dof: 1 = tASNE, Inf = ASNE
htsne <- function(dof = 1, eps = .Machine$double.eps, opt_iter = 0,
                  verbose = FALSE) {
  dof <- base::min(dof, 1e8)
  lreplace(
    asne_plugin(eps = eps),
    kernel = itsne_kernel(dof = dof),
    update_out_fn = make_update_out(keep = c("qm", "wm", "d2m")),
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
# dof: 1 = tASNE, Inf = ASNE
itsne <- function(dof = 1, opt_iter = 0, eps = .Machine$double.eps,
                  verbose = TRUE) {
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

