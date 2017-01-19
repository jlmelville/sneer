# Uses the generic "plug in" gradient as given by Lee at al.

# Factory Function Using the Plugin Gradient of Lee and co-workers
#
# An embedding method factory function.
#
# Lee and co-workers derived a generic gradient for probability-based
# embeddings in the appendix of their JSE paper. You need to provide two
# things:
#
# \enumerate{
#   \item{The gradient of the cost function with respect to the probability.}
#   \item{The gradient of the weight with respect to the squared distances.}
# }
#
# In return, you can now get the gradient of the cost function with respect
# to the embedding coordinates without having to deal with pesky chain rules
# for partial differentiation. In particular, you don't have to write your
# own definition of the stiffness function.
#
# There is a catch: depending on your definition of the weight kernel function
# and cost function, the plugin gradient could simplify to a much less complex
# expression. So the plugin gradient might be a lot slower (and less
# numerically accurate).
#
# @param cost Cost for this embedding method.
# @param kernel Similarity kernel for weight generation.
# @param stiffness_fn Stiffness function appropriate for a plugin.
# @param update_out_fn Function to run when embedding coordinates are updated.
# @param inp_updated_fn Optional custom function to run when the input data
# changes (e.g. if input perplexities have changed).
#  \code{update_out_fn} runs.
# @param out_updated_fn Optional custom function to run after
#  \code{update_out_fn} runs.
# @param prob_type Type of probability matrix used by the probability matrix,
#  e.g. "joint" or "row".
# @param after_init_fn Optional function to run after initialization has
#  occurred.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
# Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
# dimensionality reduction based on similarity preservation.
# \emph{Neurocomputing}, \emph{112}, 92-108.
# @family sneer embedding methods
# @family sneer probability embedding methods
plugin <- function(cost = kl_fg(),
                   kernel = exp_kernel(),
                   stiffness_fn = plugin_stiffness,
                   update_out_fn = make_update_out(keep = c("qm", "wm", "d2m")),
                   inp_updated_fn = NULL,
                   out_updated_fn = NULL,
                   after_init_fn = NULL,
                   prob_type = "joint",
                   eps = .Machine$double.eps,
                   verbose = TRUE) {
  method <- remove_nulls(
    list(
      cost = cost,
      kernel = kernel,
      stiffness_fn = stiffness_fn,
      out_updated_fn = out_updated_fn,
      update_out_fn = update_out_fn,
      prob_type = prob_type,
      eps = eps,
      verbose = verbose
    )
  )
  method <- on_inp_updated(method, inp_updated_fn)$method
  method
}

# ASNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of ASNE using the plugin gradient.
#
# @param beta Decay constant of the exponential similarity kernel function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{asne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
asne_plugin <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  plugin(
    cost = kl_fg(),
    kernel = exp_kernel(beta = beta),
    eps = eps,
    prob_type = "row",
    verbose = verbose
  )
}

# SSNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of SSNE using the plugin gradient.
#
# @param beta Decay constant of the exponential similarity kernel function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{ssne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
ssne_plugin <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    asne_plugin(eps = eps, beta = beta, verbose = verbose),
    prob_type = "joint"
  )
}

# t-SNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of t-SNE using the plugin gradient.
#
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{tsne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
tsne_plugin <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    ssne_plugin(eps = eps, verbose = verbose),
    kernel = tdist_kernel()
  )
}

# HSSNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of HSSNE using the plugin gradient.
#
# @param beta Decay parameter of the kernel similarity function. Equivalent
#  to the exponential decay parameter when \code{alpha} approaches zero.
# @param alpha Tail heaviness of the kernel similarity function. Must be
# greater than zero. When set to a small value this method is equivalent to
# SSNE. When set to one to one, this method behaves like t-SNE.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{hssne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
hssne_plugin <- function(beta = 1, alpha = 0, eps = .Machine$double.eps,
                         verbose = TRUE) {
  lreplace(
    ssne_plugin(eps = eps),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha),
    verbose = verbose
  )
}

# t-ASNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of t-ASNE using the plugin gradient.
#
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{tasne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
tasne_plugin <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    tsne_plugin(eps = eps, verbose = verbose),
    prob_type = "row"
  )
}

# t-PSNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of t-PSNE using the plugin gradient.
tpsne_plugin <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    tsne_plugin(eps = eps),
    prob_type = "cond",
    verbose = verbose
  )
}

# RASNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of RASNE using the plugin gradient.
#
# @param beta Decay constant of the exponential similarity kernel
#  function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{rasne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
rasne_plugin <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    asne_plugin(beta = beta, eps = eps, verbose = verbose),
    cost = reverse_kl_fg()
  )
}

# An implementation of RSSNE using the plugin gradient.
#
# @param beta Decay constant of the exponential similarity kernel function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{rssne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
rssne_plugin <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    rasne_plugin(beta = beta, eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# rt-SNE Method using Plugin Gradient
#
# A probability-based embedding method.
#
# An implementation of rt-SNE using the plugin gradient.
#
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{tsne} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
rtsne_plugin <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    rssne_plugin(eps = eps, verbose = verbose),
    kernel = tdist_kernel()
  )
}

# An implementation of NeRV using the plugin gradient.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1).
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{nerv} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
nerv_plugin <- function(lambda = 0.5, eps = .Machine$double.eps, verbose = TRUE) {
  method <- lreplace(
    asne_plugin(eps = eps, verbose = verbose),
    cost = nerv_fg(lambda = lambda)
  )
  method <- on_inp_updated(method, nerv_inp_update)$method
  method
}

# An implementation of SNeRV using the plugin gradient.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1).
#   Must be a value between 0 and 1.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{nerv} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
snerv_plugin <- function(lambda = 0.5, eps = .Machine$double.eps,
                         verbose = TRUE) {
  lreplace(
    nerv_plugin(lambda = lambda, eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# An implementation of HSNeRV using the plugin gradient.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). Must be
#   a value between 0 and 1.
# @param alpha Tail heaviness. Must be greater than zero. When set to a small
#   value this method is equivalent to SSNE or SNeRV (depending on the value
#   of \code{lambda}. When set to one to one, this method behaves like
#   t-SNE/t-NeRV.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{hsnerv} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
hsnerv_plugin <- function(lambda = 0.5, alpha = 0, eps = .Machine$double.eps,
                          verbose = TRUE) {
  lreplace(
    snerv_plugin(lambda = lambda, eps = eps, verbose = verbose),
    kernel = heavy_tail_kernel(alpha = alpha)
  )
}

# An implementation of t-NeRV using the plugin gradient.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{tnerv} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
tnerv_plugin <- function(lambda = 0.5, eps = .Machine$double.eps,
                         verbose = TRUE) {
  lreplace(
    tsne_plugin(eps = eps, verbose = verbose),
    cost = nerv_fg(lambda = lambda)
  )
}

# An implementation of UNeRV using the plugin gradient.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to ASNE. Must be a value between 0 and 1.
# @param beta Decay constant of the exponential similarity kernel function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{unerv} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
unerv_plugin <- function(lambda = 0.5, beta = 1, eps = .Machine$double.eps,
                         verbose = TRUE) {
  lreplace(
    asne_plugin(beta = beta, eps = eps, verbose = verbose),
    cost = nerv_fg(lambda = lambda)
  )
}

# An implementation of USNeRV using the plugin gradient.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1).
#   Must be a value between 0 and 1.
# @param beta Decay constant of the exponential similarity kernel function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{usnerv} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
usnerv_plugin <- function(lambda = 0.5, beta = 1, eps = .Machine$double.eps,
                          verbose = TRUE) {
  lreplace(
    unerv_plugin(lambda = lambda, beta = beta, eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# An implementation of UHSNeRV using the plugin gradient.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). Must be
#   a value between 0 and 1.
# @param beta Decay constant of the kernel similarity function. Becomes
# equivalent to the exponential decay constant as \code{alpha} approaches zero.
# @param alpha Tail heaviness of the kernel similarity function. Must be
# greater than zero. When set to a small value this method is equivalent to
# SSNE. When set to one to one, this method behaves like t-SNE.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{uhsnerv} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
uhsnerv_plugin <- function(lambda = 0.5, beta = 1, alpha = 0,
                           eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    tnerv_plugin(lambda = lambda, eps = eps, verbose = verbose),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha)
  )
}

# An implementation of JSE using the plugin gradient.
#
# @param kappa Mixture parameter. If set to 0, then JSE behaves like ASNE. If
#  set to 1, then JSE behaves like RASNE.
# @param beta Decay constant of the exponential similarity kernel function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{jse} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
jse_plugin <- function(kappa = 0.5, beta = 1, eps = .Machine$double.eps,
                       verbose = TRUE) {
  lreplace(
    asne_plugin(beta = beta, eps = eps, verbose = verbose),
    cost = jse_fg(kappa = kappa),
    out_updated_fn = klqz_update
  )
}

# An implementation of SJSE using the plugin gradient.
#
# @param kappa Mixture parameter. If set to 0, then JSE behaves like SSNE. If
#  set to 1, then JSE behaves like RSSNE.
# @param beta Decay constant of the exponential similarity kernel function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{sjse} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
sjse_plugin <- function(kappa = 0.5, beta = 1, eps = .Machine$double.eps,
                        verbose = TRUE) {
  lreplace(
    jse_plugin(kappa = kappa, beta = beta, eps = eps, verbose = verbose),
    prob_type = "joint"
  )
}

# An implementation of HSJSE using the plugin gradient.
#
# @param kappa Mixture parameter. If set to 0, then JSE behaves like SSNE. If
#  set to 1, then JSE behaves like RSSNE.
# @param beta Decay constant of the similarity kernel. Equivalent to the
#  exponential decay constant as \code{alpha} approaches zero.
# @param alpha Tail heaviness of the weighting function.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{sjse} should give equivalent results, but is probably
# a bit more efficient.
# @family sneer embedding methods
# @family sneer probability embedding methods
hsjse_plugin <- function(kappa = 0.5, beta = 1, alpha = 0,
                         eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    sjse_plugin(kappa = kappa, eps = eps, verbose = verbose),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha)
  )
}

# Plugin Stiffness
#
# Calculates the Stiffness matrix of an embedding method using the plugin
# gradient formulation.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Stiffness matrix.
plugin_stiffness <- function(method, inp, out) {
  prob_type <- method$prob_type

  fn_name <- paste0('plugin_stiffness_', prob_type)
  stiffness_fn <- get(fn_name)
  if (is.null(stiffness_fn)) {
    stop("Unable to find plugin stiffness function for ", prob_type)
  }
  stiffness_fn(method, inp, out)
}

# Plugin Stiffness for Row Probabilities
#
# Calculates the stiffness matrix for row probability based embedding methods.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Stiffness matrix.
plugin_stiffness_row <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)
  dw_df <- method$kernel$gr(method$kernel, out$d2m)

  wm_sum <-  rowSums(out$wm)
  km <- rowSums(dc_dq * out$qm)
  km <- sweep(dc_dq, 1, km) # subtract row sum from each row element
  km <- km * (dw_df / wm_sum)
  2 * (km + t(km))
}

# Plugin Stiffness for Conditional Probabilities
#
# Calculates the stiffness matrix for conditional probability based embedding
# methods.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Stiffness matrix.
plugin_stiffness_cond <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)
  dw_du <- method$kernel$gr(method$kernel, out$d2m)
  wm_sum <- sum(out$wm)
  km <- (dc_dq - sum(dc_dq * out$qm)) * (dw_du / wm_sum)
  2 * (km + t(km))
}

# Plugin Stiffness for Joint Probabilities
#
# The stiffness matrix is identical for joint and conditional P matrices, because
# they both sum over all pairs, rather than all points.
#
# Further simplification for joint probability embedding is possible only if Q
# is joint as well as P (by either enforcing jointness as done with P, or
# because the similarity kernel is naturally symmetric). In that case, we
# could replace 2 * (km * t(km)) with 4 * km (but we don't support that yet).
plugin_stiffness_joint <- plugin_stiffness_cond
