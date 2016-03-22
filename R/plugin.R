# Uses the generic "plug in" gradient as given by Lee at al.

#' Factory Function Using the Plugin Gradient of Lee and co-workers
#'
#' An embedding method factory function.
#'
#' Lee and co-workers derived a generic gradient for probability-based
#' embeddings in the appendix of their JSE paper. You need to provide two
#' things:
#'
#' \enumerate{
#'   \item{The gradient of the cost function with respect to the probability.}
#'   \item{The gradient of the weight with respect to the squared distances.}
#' }
#'
#' In return, you can now get the gradient of the cost function with respect
#' to the embedding coordinates without having to deal with pesky chain rules
#' for partial differentiation. In particular, you don't have to write your
#' own definition of the stiffness function.
#'
#' There is a catch: depending on your definition of the weight kernel function
#' and cost function, the plugin gradient could simplify to a much less complex
#' expression. So the plugin gradient might be a lot slower (and less
#' numerically accurate).
#'
#' @param cost Cost for this embedding method.
#' @param kernel Similarity kernel for weight generation.
#' @param stiffness_fn Stiffness function appropriate for a plugin.
#' @param update_out_fn Function to run when embedding coordinates are updated.
#' @param out_updated_fn Optional custom function to run after
#'  \code{update_out_fn} runs.
#' @param prob_type Type of probability matrix used by the probability matrix,
#'  e.g. "joint" or "row".
#' @param after_init_fn Optional function to run after initialization has
#'  occurred.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
plugin <- function(cost = kl_fg(),
                   kernel = exp_kernel(),
                   stiffness_fn = plugin_stiffness,
                   update_out_fn = update_out(keep = c("qm", "wm", "d2m")),
                   inp_updated_fn = NULL,
                   out_updated_fn = NULL,
                   after_init_fn = NULL,
                   prob_type = "joint",
                   eps = .Machine$double.eps) {
  remove_nulls(
    list(
      cost = cost,
      kernel = kernel,
      stiffness_fn = stiffness_fn,
      inp_updated_fn = inp_updated_fn,
      out_updated_fn = out_updated_fn,
      update_out_fn = update_out_fn,
      prob_type = prob_type,
      eps = eps
    )
  )
}

#' ASNE Method using Plugin Gradient
#'
#' A probability-based embedding method.
#'
#' An implementation of ASNE using the plugin gradient.
#'
#' @param beta Precision parameter of the exponential similarity kernel
#'  function.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{asne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
asne_plugin <- function(beta = 1, eps = .Machine$double.eps) {
  plugin(
    cost = kl_fg(),
    kernel = exp_kernel(beta = beta),
    eps = eps,
    prob_type = "row"
  )
}

#' SSNE Method using Plugin Gradient
#'
#' A probability-based embedding method.
#'
#' An implementation of SSNE using the plugin gradient.
#'
#' @param beta Precision parameter of the exponential similarity kernel
#'  function.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{ssne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
ssne_plugin <- function(eps = .Machine$double.eps, beta = 1) {
  lreplace(
    asne_plugin(eps = eps, beta = beta),
    prob_type = "joint"
  )
}

#' t-SNE Method using Plugin Gradient
#'
#' A probability-based embedding method.
#'
#' An implementation of t-SNE using the plugin gradient.
#'
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{tsne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
tsne_plugin <- function(eps = .Machine$double.eps) {
  lreplace(
    ssne_plugin(eps = eps),
    kernel = tdist_kernel()
  )
}

#' HSSNE Method using Plugin Gradient
#'
#' A probability-based embedding method.
#'
#' An implementation of HSSNE using the plugin gradient.
#'
#' @param alpha Tail heaviness of the kernel similarity function. Must be
#' greater than zero. When set to a small value this method is equivalent to
#' SSNE. When set to one to one, this method behaves like t-SNE.
#' @param beta The precision of the kernel similarity function. Becomes
#' equivalent to the precision in the Gaussian distribution of distances as
#' \code{alpha} approaches zero.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{hssne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
hssne_plugin <- function(beta = 1, alpha = 0, eps = .Machine$double.eps) {
  lreplace(
    ssne_plugin(eps = eps),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha)
  )
}

#' t-ASNE Method using Plugin Gradient
#'
#' A probability-based embedding method.
#'
#' An implementation of t-ASNE using the plugin gradient.
#'
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{tasne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
tasne_plugin <- function(eps = .Machine$double.eps) {
  lreplace(
    tsne_plugin(eps = eps),
    prob_type = "row"
  )
}

#' RASNE Method using Plugin Gradient
#'
#' A probability-based embedding method.
#'
#' An implementation of RASNE using the plugin gradient.
#'
#' @param beta Precision parameter of the exponential similarity kernel
#'  function.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{rasne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
rasne_plugin <- function(beta = 1, eps = .Machine$double.eps) {
  lreplace(
    asne_plugin(beta = beta, eps = eps),
    cost = reverse_kl_fg(),
    out_updated_fn = klqp_update
  )
}

#' An implementation of RSSNE using the plugin gradient.
#'
#' @param beta Precision parameter of the exponential similarity kernel
#'  function.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{rssne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
rssne_plugin <- function(eps = .Machine$double.eps, beta = 1) {
  lreplace(
    rasne_plugin(beta = beta, eps = eps),
    prob_type = "joint"
  )
}

#' rt-SNE Method using Plugin Gradient
#'
#' A probability-based embedding method.
#'
#' An implementation of rt-SNE using the plugin gradient.
#'
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{tsne}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
rtsne_plugin <- function(eps = .Machine$double.eps) {
  lreplace(
    rssne_plugin(eps = eps),
    kernel = tdist_kernel()
  )
}

#' An implementation of NeRV using the plugin gradient.
#'
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to ASNE. Must be a value between 0 and 1.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{nerv}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
nerv_plugin <- function(lambda = 0.5, eps = .Machine$double.eps) {
  lreplace(
    asne_plugin(eps = eps),
    cost = nerv_fg(lambda = lambda),
    kernel = exp_kernel(),
    inp_updated_fn = transfer_kernel_betas,
    out_updated_fn = klqp_update
  )
}

#' An implementation of SNeRV using the plugin gradient.
#'
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to SSNE. Must be a value between 0 and 1.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{nerv}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
snerv_plugin <- function(lambda = 0.5, eps = .Machine$double.eps) {
  lreplace(
    nerv_plugin(lambda = lambda, eps = eps),
    prob_type = "joint"
  )
}

#' An implementation of HSNeRV using the plugin gradient.
#'
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
#' @param alpha Tail heaviness. Must be greater than zero. When set to a small
#'   value this method is equivalent to SSNE or SNeRV (depending on the value
#'   of \code{lambda}. When set to one to one, this method behaves like
#'   t-SNE/t-NeRV.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{hsnerv}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
hsnerv_plugin <- function(lambda = 0.5, alpha = 0, eps = .Machine$double.eps) {
  lreplace(
    snerv_plugin(lambda = lambda, eps = eps),
    kernel = heavy_tail_kernel(alpha = alpha)
  )
}

#' An implementation of t-NeRV using the plugin gradient.
#'
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{tnerv}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
tnerv_plugin <- function(lambda = 0.5, eps = .Machine$double.eps) {
  lreplace(
    tsne_plugin(eps = eps),
    cost = nerv_fg(lambda = lambda),
    out_updated_fn = klqp_update
  )
}

#' An implementation of UNeRV using the plugin gradient.
unerv_plugin <- function(lambda = 0.5, beta = 1, eps = .Machine$double.eps) {
  lreplace(
    asne_plugin(beta = beta, eps = eps),
    cost = nerv_fg(lambda = lambda),
    kernel = exp_kernel(),
    out_updated_fn = klqp_update
  )
}

#' An implementation of USNeRV using the plugin gradient.
usnerv_plugin <- function(lambda = 0.5, beta = 1, eps = .Machine$double.eps) {
  lreplace(
    unerv_plugin(lambda = lambda, beta = beta, eps = eps),
    prob_type = "joint"
  )
}

#' An implementation of UHSNeRV using the plugin gradient.
uhsnerv_plugin <- function(lambda = 0.5, beta = 1, alpha = 0,
                           eps = .Machine$double.eps) {
  lreplace(
    tnerv_plugin(lambda = lambda, eps = eps),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha)
  )
}

#' An implementation of JSE using the plugin gradient.
#'
#' @param kappa Mixture parameter. If set to 0, then JSE behaves like ASNE. If
#'  set to 1, then JSE behaves like RASNE.
#' @param beta Precision parameter of the exponential similarity kernel
#'  function.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{jse}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
jse_plugin <- function(kappa = 0.5, beta = 1, eps = .Machine$double.eps) {
  lreplace(
    asne_plugin(beta = beta, eps = eps),
    cost = jse_fg(kappa = kappa),
    out_updated_fn = klqz_update
  )
}

#' An implementation of SJSE using the plugin gradient.
#'
#' @param kappa Mixture parameter. If set to 0, then JSE behaves like SSNE. If
#'  set to 1, then JSE behaves like RSSNE.
#' @param beta Precision parameter of the exponential similarity kernel
#'  function.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{sjse}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
sjse_plugin <- function(kappa = 0.5, beta = 1, eps = .Machine$double.eps) {
  lreplace(
    jse_plugin(kappa = kappa, beta = beta, eps = eps),
    prob_type = "joint"
  )
}

#' An implementation of HSJSE using the plugin gradient.
#'
#' @param kappa Mixture parameter. If set to 0, then JSE behaves like SSNE. If
#'  set to 1, then JSE behaves like RSSNE.
#' @param beta Precision parameter of the exponential similarity kernel
#'  function.
#' @param alpha Tail heaviness of the weighting function.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return An embedding method for use by an embedding function.
#' @seealso \code{\link{sjse}} should give equivalent results, but is probably
#' a bit more efficient.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
hsjse_plugin <- function(kappa = 0.5, beta = 1, alpha = 0,
                         eps = .Machine$double.eps) {
  lreplace(
    sjse_plugin(kappa = kappa, eps = eps),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha)
  )
}

#' Plugin Stiffness
#'
#' Calculates the Stiffness matrix of an embedding method using the plugin
#' gradient formulation.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Stiffness matrix.
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
#'
#' Calculates the stiffness matrix for row probability based embedding methods.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Stiffness matrix.
plugin_stiffness_row <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)
  dw_du <- method$kernel$gr(method$kernel, out$d2m)

  wm_sum <-  apply(out$wm, 1, sum)
  km <- apply(dc_dq * out$qm, 1, sum) # row sums
  km <- sweep(-dc_dq, 1, -km) # subtract row sum from each row element
  km <- km * (-dw_du / wm_sum)
  2 * (km + t(km))
}

#' Plugin Stiffness for Joint Probabilities
#'
#' Calculates the stiffness matrix for joint probability based embedding
#' methods.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Stiffness matrix.
plugin_stiffness_joint <- function(method, inp, out) {
  wm_sum <- sum(out$wm)
  dc_dq <- method$cost$gr(inp, out, method)
  dw_du <- method$kernel$gr(method$kernel, out$d2m)
  km <- (sum(dc_dq * out$qm) - dc_dq) * (-dw_du / wm_sum)
  2 * (km + t(km))
}

