# Uses the generic "plug in" gradient as given by Lee at al.

#'
plugin <- function(cost = kl(),
                   kernel = exp_kernel(),
                   stiffness_fn = plugin_stiffness,
                   update_out_fn = update_out(keep = c("qm", "wm", "d2m")),
                   prob_type = "joint",
                   eps = .Machine$double.eps,
                   beta = 1){
  list(
    cost_fn = cost$fn,
    cost_gr = cost$gr,
    kernel = kernel,
    weight_fn = kernel$fn,
    beta = beta,
    stiffness_fn = stiffness_fn,
    update_out_fn = update_out_fn,
    prob_type = prob_type,
    eps = eps
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
hssne_plugin <- function(eps = .Machine$double.eps, beta = 1, alpha = 0) {
  plugin(
    cost = kl(),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha),
    eps = eps,
    prob_type = "joint"
  )
}


#' KL
kl <- function() {
  list(
    fn = kl_cost,
    gr = kl_cost_gr
  )
}

#' Gradient Wrapper
kl_cost_gr <- function(inp, out, method) {
  kl_divergence_gr(inp$pm, out$qm, method$eps)
}

#' KL Gradient
kl_divergence_gr <- function(pm, qm, eps = .Machine$double.eps) {
  -pm / (qm + eps)
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

