# Uses the generic "plug in" gradient as given by Lee at al.

#'
plugin <- function(cost = kl(),
                   kernel = exp_kernel(),
                   stiffness_fn = plugin_stiffness,
                   update_out_fn = update_out(keep = c("qm", "wm", "d2m")),
                   out_updated_fn = NULL,
                   prob_type = "joint",
                   eps = .Machine$double.eps) {
  remove_nulls(
    list(
      cost = cost,
      cost_fn = cost$fn,
      cost_gr = cost$gr,
      kernel = kernel,
      weight_fn = kernel$fn,
      stiffness_fn = stiffness_fn,
      out_updated_fn = out_updated_fn,
      update_out_fn = update_out_fn,
      prob_type = prob_type,
      eps = eps
    )
  )
}

#' SSNE Plugin
ssne_plugin <- function(eps = .Machine$double.eps, beta = 1) {
  plugin(
    cost = kl(),
    kernel = exp_kernel(beta = beta),
    eps = eps,
    prob_type = "joint"
  )
}

#' ASNE plugin
asne_plugin <- function(eps = .Machine$double.eps, beta = 1) {
  plugin(
    cost = kl(),
    kernel = exp_kernel(beta = beta),
    eps = eps,
    prob_type = "row"
  )
}

#' TSNE plugin
tsne_plugin <- function(eps = .Machine$double.eps) {
  plugin(
    cost = kl(),
    kernel = tdist_kernel(),
    eps = eps,
    prob_type = "joint"
  )
}

#' HSSNE plugin
hssne_plugin <- function(beta = 1, alpha = 0, eps = .Machine$double.eps) {
  plugin(
    cost = kl(),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha),
    eps = eps,
    prob_type = "joint"
  )
}

#' TASNE plugin
tasne_plugin <- function(eps = .Machine$double.eps) {
  plugin(
    cost = kl(),
    kernel = tdist_kernel(),
    eps = eps,
    prob_type = "row"
  )
}

#' RASNE plugin
rasne_plugin <- function(eps = .Machine$double.eps, beta = 1) {
  plugin(
    cost = reverse_kl(),
    kernel = exp_kernel(beta = beta),
    out_updated_fn = klqp_update,
    eps = eps,
    prob_type = "row"
  )
}

#' RSSNE Plugin
rssne_plugin <- function(eps = .Machine$double.eps, beta = 1) {
  plugin(
    cost = reverse_kl(),
    kernel = exp_kernel(beta = beta),
    out_updated_fn = klqp_update,
    eps = eps,
    prob_type = "joint"
  )
}

#' RTSNE plugin
rtsne_plugin <- function(eps = .Machine$double.eps) {
  plugin(
    cost = reverse_kl(),
    kernel = tdist_kernel(),
    out_updated_fn = klqp_update,
    eps = eps,
    prob_type = "joint"
  )
}

#' NeRV
nerv_plugin <- function(beta = 1, lambda = 0.5, eps = .Machine$double.eps) {
  plugin(
    cost = nerv_fg(lambda = lambda),
    kernel = exp_kernel(beta = beta),
    out_updated_fn = klqp_update,
    prob_type = "row",
    eps = eps
  )
}

snerv_plugin <- function(beta = 1, lambda = 0.5, eps = .Machine$double.eps) {
  lreplace(
    nerv_plugin(lambda = lambda, beta = beta, eps = eps),
    prob_type = "joint"
  )
}

tnerv_plugin <- function(lambda = 0.5, eps = .Machine$double.eps) {
  lreplace(
    snerv_plugin(lambda = lambda, beta = beta, eps = eps),
    kernel = tdist_kernel(),
    weight_fn = tdist_kernel()$fn
  )
}

hsnerv_plugin <- function(lambda = 0.5, beta = 1, alpha = 0,
                          eps = .Machine$double.eps) {
  lreplace(
    snerv_plugin(lambda = lambda, beta = beta, eps = eps),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha),
    weight_fn = heavy_tail_kernel(beta = beta, alpha = alpha)$fn
  )
}

#' Plugin Stiffness
plugin_stiffness <- function(method, inp, out) {
  prob_type <- method$prob_type
  if (is.null(prob_type)) {
    stop("Embedding method must have a prob type")
  }
  fn_name <- paste0('plugin_stiffness_', prob_type)
  stiffness_fn <- get(fn_name)
  if (is.null(stiffness_fn)) {
    stop("Unable to find plugin stiffness function for ", prob_type)
  }
  stiffness_fn(method, inp, out)
}

#' Plugin Stiffness for Row Probabilities
plugin_stiffness_row <- function(method, inp, out) {
  cm_grad <- method$cost_gr(inp, out, method)
  wm_grad <- method$kernel$gr(out$d2m)

  wm_sum <-  apply(out$wm, 1, sum)
  km <- apply(cm_grad * out$qm, 1, sum) # row sums
  km <- sweep(-cm_grad, 1, -km) # subtract row sum from each row element
  km <- km * (-wm_grad / wm_sum)
  2 * (km + t(km))
}

#' Plugin Stiffness for Joint Probabilities
plugin_stiffness_joint <- function(method, inp, out) {
  wm_sum <- sum(out$wm)
  cm_grad <- method$cost_gr(inp, out, method)
  wm_grad <- method$kernel$gr(out$d2m)
  km <- (sum(cm_grad * out$qm) - cm_grad) * (-wm_grad / wm_sum)
  2 * (km + t(km))
}

