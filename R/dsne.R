# A set of methods where the meta-parameters are also optimized.
# As these fluctuate over the course of the optimization, I call them
# "dynamic" SNE methods.

dhssne <- function(beta = 1, alpha = 0, eps = .Machine$double.eps,
                   verbose = TRUE) {
  lreplace(
    hssne(beta = beta, alpha = alpha, eps = eps, verbose = verbose),
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
    extra_gr = function(opt, inp, out, method, extra_par) {
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
    }
  )
}
