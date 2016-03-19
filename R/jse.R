#' Jensen-Shannon Embedding (JSE)
#'
#' A probability-based embedding method.
#'
#' JSE is a variant of Asymmetric Stochastic Neighbor Embedding
#' (see \code{\link{asne}}), with a modified cost function that uses the
#' a slightly modified version of the Generalized Jensen-Shannon Divergence,
#' rather than the Kullback-Leibler divergence. The JS divergence can be
#' considered a symmetrized and smoothed version of the KL divergence.
#'
#' The JSE cost function modifies the JS divergence to allow the degree of
#' symmetry in the divergence between two probability distributions, P and Q, to
#' be controlled by a parameter, kappa, which takes a value between 0 and 1
#' (exclusive). At its default value of 0.5, it reproduces the symmetric
#' JS divergence. As kappa approaches zero, its behavior approaches that of the
#' KL divergence, KL(P||Q) (and hence ASNE). As kappa aproaches one, its
#' behaviour approaches that of the "reverse" KL divergence, KL(Q||P)
#' (and hence like \code{\link{rasne}}). You won't get exactly identical results
#' to RASNE and ASNE, because of numerical issues.
#'
#' The probability matrix used in JSE:
#'
#' \itemize{
#'  \item{represents one N row-wise probability distributions, where N is the
#'  number of points in the data set, i.e. the row sums of the matrix are all
#'   one.}
#'  \item{is asymmetric, i.e. there is no requirement that
#'  \code{p[i, j] == p[j, i]}.}
#' }
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#'  \code{wm}.}
#' }
#' @param kappa Mixture parameter. If set to 0, then JSE behaves like ASNE. If
#'  set to 1, then JSE behaves like RASNE.
#' @param beta The precision of the weighting function.
#' @param eps Small floating point value used to prevent numerical problems,
#'   e.g. in gradients and cost functions.
#' @param verbose If \code{TRUE}, log information about the embedding.
#' @return An embedding method for use by an embedding function.
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#' @seealso JSE uses the \code{\link{jse_cost}} cost function and the
#'   \code{\link{exp_weight}} similarity function for converting distances to
#'   probabilities. The \code{\link{nerv}} embedding method also uses a cost
#'   function which is the sum of KL divergences, controlled by a parameter,
#'   and which also reduces to ASNE at one extreme, and to "reverse" ASNE at
#'   another.
#' The return value of this function should be used with the
#' \code{\link{embed_prob}} embedding function.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @examples
#' \dontrun{
#' # default JSE, cost function is symmetric
#' embed_prob(method = jse(kappa = 0.5), ...)
#'
#' # equivalent to ASNE
#' embed_prob(method = jse(kappa = 0), ...)
#'
#' # equivalent to "reverse" ASNE
#' embed_prob(method = jse(kappa = 1), ...)
#' }
jse <- function(kappa = 0.5, beta = 1, eps = .Machine$double.eps,
                verbose = TRUE) {
  lreplace(
    asne(beta = beta, eps = eps, verbose = verbose),
    cost = jse_fg(kappa = kappa),
    cost_fn = jse_fg(kappa = kappa)$fn,
    stiffness_fn = function(method, inp, out) {
      jse_stiffness(out$qm, out$zm, out$kl_qz, kappa = method$cost$kappa,
                    beta = method$kernel$beta, eps = method$eps)
    },
    out_updated_fn = klqz_update,
    inp_updated = function(inp, out, method) {
      if (!is.null(out)) {
        out <- method$update_out_fn(inp, out, method)
      }
      list(out = out)
    },
    kappa = clamp(kappa, min_val = sqrt(.Machine$double.eps),
                  max_val = 1 - sqrt(.Machine$double.eps))
  )
}

#' Symmetric Jensen-Shannon Embedding (SJSE)
#'
#' A probability-based embedding method.
#'
#' SJSE is a variant of \code{\link{jse}} which uses a symmetrized, normalized
#' probability distribution like \code{\link{ssne}}, rather than the that used
#' by the original JSE method, which used the unnormalized distributions of
#' \code{\link{asne}}.
#'
#' The probability matrix used in SJSE:
#'
#' \itemize{
#'  \item{represents one probability distribution, i.e. the grand sum of the
#'  matrix is one.}
#'  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#'  probabilities are joint probabilities.}
#' }
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#'  \code{wm}.}
#' }
#' @param kappa Mixture parameter. Cost function behaves more like the
#'   Kullback-Leibler divergence as it approaches zero and more like the
#'   "reverse" KL divergence as it approaches one.
#' @param beta The precision of the weighting function.
#' @param eps Small floating point value used to prevent numerical problems,
#'   e.g. in gradients and cost functions.
#' @param verbose If \code{TRUE}, log information about the embedding.
#' @return An embedding method for use by an embedding function.
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#' @seealso SJSE uses the \code{\link{jse_cost}} cost function and the
#'   \code{\link{exp_weight}} similarity function for converting
#'   distances to probabilities. The \code{\link{snerv}} embedding method is
#'   similar.
#' The return value of this function should be used with the
#' \code{\link{embed_prob}} embedding function.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @examples
#' \dontrun{
#' # default SJSE, cost function is symmetric
#' embed_prob(method = hsjse(kappa = 0.5), ...)
#'
#' # equivalent to SSNE
#' embed_prob(method = hsjse(kappa = 0), ...)
#'
#' # equivalent to "reverse" SSNE
#' embed_prob(method = hsjse(kappa = 1), ...)
#' }
sjse <- function(kappa = 0.5, beta = 1, eps = .Machine$double.eps,
                 verbose = TRUE) {
  lreplace(
    jse(kappa = kappa, beta = beta, eps = eps, verbose = verbose),
    stiffness_fn = function(method, inp, out) {
      sjse_stiffness(out$qm, out$zm, out$kl_qz, kappa = method$cost$kappa,
                     beta = method$kernel$beta, eps = method$eps)
    },
    prob_type = "joint"
  )
}

#' Heavy-Tailed Symmetric Jensen-Shannon Embedding (HSJSE)
#'
#' A probability-based embedding method.
#'
#' HSJSE is a variant of \code{\link{jse}} which uses a symmetrized, normalized
#' probability distribution like \code{\link{ssne}}, rather than the that used
#' by the original JSE method, which used the unnormalized distributions of
#' \code{\link{asne}}.
#'
#' Additionally, it uses the heavy-tailed kernel function of
#' \code{\link{hssne}}, to generalize exponential and t-distributed weighting.
#' By modifying the \code{alpha} and \code{kappa} parameters, this embedding
#' method can reproduce multiple embedding methods (see the examples section).
#'
#' The probability matrix used in HSJSE:
#'
#' \itemize{
#'  \item{represents one probability distribution, i.e. the grand sum of the
#'  matrix is one.}
#'  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#'  probabilities are joint probabilities.}
#' }
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#'  \code{wm}.}
#' }
#' @param kappa Mixture parameter. Cost function behaves more like the
#'   Kullback-Leibler divergence as it approaches zero and more like the
#'   "reverse" KL divergence as it approaches one.
#' @param alpha Tail heaviness. Must be greater than zero. Set to zero for
#'   a Gaussian-like kernel, and to one for a Student-t distribution.
#' @param beta The precision of the function. Becomes equivalent to the
#'   precision in the Gaussian distribution of distances as \code{alpha}
#'   approaches zero.
#' @param eps Small floating point value used to prevent numerical problems,
#'   e.g. in gradients and cost functions.
#' @param verbose If \code{TRUE}, log information about the embedding.
#' @return An embedding method for use by an embedding function.
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
#' Heavy-tailed symmetric stochastic neighbor embedding.
#' In \emph{Advances in neural information processing systems} (pp. 2169-2177).
#' @seealso HSJSE uses the \code{\link{jse_cost}} cost function and the
#'   \code{\link{heavy_tail_weight}} similarity function for converting
#'   distances to probabilities. The \code{\link{hsnerv}} embedding method is
#'   similar.
#' The return value of this function should be used with the
#' \code{\link{embed_prob}} embedding function.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @examples
#' \dontrun{
#' # default HSJSE, cost function is symmetric
#' embed_prob(method = hsjse(kappa = 0.5), ...)
#'
#' # equivalent to SSNE
#' embed_prob(method = hsjse(kappa = 0, alpha = 0), ...)
#'
#' # equivalent to "reverse" SSNE
#' embed_prob(method = hsjse(kappa = 1, alpha = 0), ...)
#'
#' # equivalent to t-SNE
#' embed_prob(method = hsjse(kappa = 0, alpha = 1), ...)
#'
#' # equivalent to "reverse" t-SNE
#' embed_prob(method = hsjse(kappa = 1, alpha = 1), ...)
#' }
hsjse <- function(kappa = 0.5, alpha = 0, beta = 1, eps = .Machine$double.eps,
                  verbose = TRUE) {
  lreplace(
    sjse(kappa = kappa, eps = eps, verbose = verbose),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha),
    stiffness_fn = function(method, inp, out) {
      hsjse_stiffness(out$qm, out$zm, out$wm, out$kl_qz,
                      kappa = method$cost$kappa, alpha = method$kernel$alpha,
                      beta = method$kernel$beta, eps = method$eps)
    },
    update_out_fn = update_out(keep = c("qm", "wm"))
  )
}

#' JSE Stiffness Function
#'
#' @param qm Output probabilty matrix.
#' @param zm Mixture matrix, weighted combination of input probability and
#'  output probability matrix \code{qm}.
#' @param kl_qz KL divergence between \code{qm} and \code{zm}. \code{qm} is the
#'  reference probability.
#' @param kappa Mixture parameter. Should be a value between 0 and 1 and be the
#'  same value used to produce the mixture matrix \code{zm}.
#' @param beta The precision of the weighting function.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
jse_stiffness <- function(qm, zm, kl_qz, kappa = 0.5, beta = 1,
                          eps = .Machine$double.eps) {
  reverse_asne_stiffness(zm, qm, kl_qz, beta = beta, eps = eps) / kappa
}

#' Symmetric JSE Stiffness Function
#'
#' @param qm Output probabilty matrix.
#' @param zm Mixture matrix, weighted combination of input probability and
#'  output probability matrix \code{qm}.
#' @param kl_qz KL divergence between \code{qm} and \code{zm}. \code{qm} is the
#'  reference probability.
#' @param kappa Mixture parameter. Should be a value between 0 and 1 and be the
#'  same value used to produce the mixture matrix \code{zm}.
#' @param beta The precision of the weighting function.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
sjse_stiffness <- function(qm, zm, kl_qz, kappa = 0.5, beta = 1,
                          eps = .Machine$double.eps) {
  reverse_ssne_stiffness(zm, qm, kl_qz, beta = beta, eps = eps) / kappa
}

#' HSJSE Stiffness Function
#'
#' @param qm Output probabilty matrix.
#' @param zm Mixture matrix, weighted combination of input probability and
#'  output probability matrix \code{qm}.
#' @param wm Output weight probability matrix.
#' @param kl_qz KL divergence between \code{qm} and \code{zm}. \code{qm} is the
#'  reference probability.
#' @param kappa Mixture parameter. Should be a value between 0 and 1 and be the
#'  same value used to produce the mixture matrix \code{zm}.
#' @param alpha Tail heaviness of the weighting function.
#' @param beta The precision of the weighting function.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
hsjse_stiffness <- function(qm, zm, wm, kl_qz, kappa = 0.5, alpha = 1.5e-8,
                            beta = 1, eps = .Machine$double.eps) {
  reverse_hssne_stiffness(zm, qm, wm, kl_qz,
                          alpha = alpha, beta = beta, eps = eps) / kappa
}

#' JSE Cost Function
#'
#' A measure of embedding quality between input and output data.
#'
#' This cost function evaluates the embedding quality by calculating the JSE
#' divergence, a variation on the generalized Jensen-Shannon divergence between
#' the input probabilities and the output probabilities. The JSE Divergence
#' between two discrete probabilities P and Q is:
#'
#' \deqn{D_{JSE}(P||Q)=\frac{1}{1-\kappa}D_{KL}(P||Z) + \frac{1}{\kappa}D_{KL}(Q||Z)}{D_JSE(P||Q) = ((1/(1-kappa))*D_KL(P||Z)) + ((1/kappa)*D_KL(Q||Z))}
#'
#' where Z is a mixture matrix of \eqn{P} and \eqn{Q}:
#'
#' \deqn{Z = \kappa P + (1 - \kappa)Q}{Z = kappa * P + (1 - kappa) * Q}
#'
#' and \eqn{D_{KL}(P||Q)}{D_KL(P||Q)} is the Kullback-Leibler divergence
#' between \eqn{P} and \eqn{Q}:
#'
#' \deqn{D_{KL}(P||Q) = \sum_{i}P(i)\log\frac{P(i)}{Q(i)}}{D_KL(P||Q) = sum(Pi*log(Pi/Qi))}
#'
#' This cost function requires the following matrices to be defined:
#' \describe{
#'  \item{\code{inp$pm}}{Input probabilities.}
#'  \item{\code{out$qm}}{Output probabilities.}
#'  \item{\code{out$zm}}{Mixture probabilities: a weighted linear combination
#'    of \code{inp$pm} and \code{out$qm}.}
#' }
#'
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return JSE divergence between \code{inp$pm} and \code{out$qm}.
#' @seealso To use \code{out$qm} as the reference probability and calculate the
#'   divergence of \code{inp$pm} from \code{out$qm}, see
#'   \code{\link{reverse_kl_cost}}.
#' @family sneer cost functions
jse_cost <- function(inp, out, method) {
  jse_divergence(inp$pm, out$qm, out$zm, method$cost$kappa, method$eps)
}
attr(jse_cost, "sneer_cost_type") <- "prob"
attr(jse_cost, "sneer_cost_norm") <- "jse_cost_norm"

#' Normalized JSE Cost Function
#'
#' A measure of embedding quality between input and output data.
#'
#' Normalizes the JSE cost using the cost when the output probability matrix
#' \code{out$qm} is uniform. Also recalculates the mixture matrix \code{out$zm}
#' too. Intended to be used in the reporter function of sneer as a custom
#' normalized cost function, not as a main objective function.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return JSE divergence between \code{inp$pm} and \code{out$qm}.
jse_cost_norm <- function(inp, out, method) {
  cost <- jse_divergence(inp$pm, out$qm, out$zm, method$cost$kappa, method$eps)
  null_qm <- null_model_prob(out$qm)
  null_zm <- js_mixture(inp$pm, null_qm, method$cost$kappa)
  null_cost <- jse_divergence(inp$pm, null_qm, null_zm, method$cost$kappa,
                              method$eps)
  cost / null_cost
}

#' Jensen-Shannon Embedding (JSE) Divergence
#'
#' A measure of embedding quality between input and output probability matrices.
#'
#' The JSE Divergence between two discrete probabilities P and Q
#' is:
#'
#' \deqn{D_{JSE}(P||Q)=\frac{1}{1-\kappa}D_{KL}(P||Z) + \frac{1}{\kappa}D_{KL}(Q||Z)}{D_JSE(P||Q) = ((1/(1-kappa))*D_KL(P||Z)) + ((1/kappa)*D_KL(Q||Z))}
#'
#' where Z is a mixture matrix of \eqn{P} and \eqn{Q}:
#'
#' \deqn{Z = \kappa P + (1 - \kappa)Q}{Z = kappa * P + (1 - kappa) * Q}
#'
#' and \eqn{D_{KL}(P||Q)}{D_KL(P||Q)} is the Kullback-Leibler divergence
#' between \eqn{P} and \eqn{Q}:
#'
#' \deqn{D_{KL}(P||Q) = \sum_{i}P(i)\log\frac{P(i)}{Q(i)}}{D_KL(P||Q) = sum(Pi*log(Pi/Qi))}
#'
#' The base of the log determines the units of the divergence.
#'
#' The JSE divergence is a variation of the Generalized Jensen-Shannon
#' Divergence for two distributions with the mixing parameter,
#' \eqn{\kappa}{kappa}, modified so that the divergence has limiting values of
#' \eqn{D_{KL}(P||Q)}{D_KL(P||Q)} and \eqn{D_{KL}(Q||P)}{D_KL(Q||P)} as
#' \eqn{\kappa}{kappa} approaches zero and one, respectively.
#'
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' @param pm Probability Matrix.
#' @param qm Probability Matrix.
#' @param zm Mixture probability matrix, composed of a weighted sum of
#'  \code{pm} and \code{qm}. If \code{NULL}, will be calculated using the
#'  provided value of \code{kappa}. If provided, the value of \code{kappa} used
#'  to generate should have be the same as the one provided to the function.
#' @param kappa Mixture parameter.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return JSE divergence between \code{pm} and \code{qm}.
jse_divergence <- function(pm, qm, zm = NULL, kappa = 0.5,
                           eps = .Machine$double.eps) {
  if (is.null(zm)) {
    zm <- js_mixture(pm, qm, kappa)
  }
  (kl_divergence(pm, zm) / (1 - kappa)) +
    (kl_divergence(qm, zm) / (kappa))
}

#' Jensen-Shannon Mixture Matrix
#'
#' Creates a mixture matrix, \eqn{Z}, comprised of a linear weighted mixture of
#' \eqn{P} and \eqn{Q}:
#'
#' \deqn{Z = \kappa P + (1 - \kappa)Q}{Z = kappa * P + (1 - kappa) * Q}
#'
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' @param pm Probability matrix.
#' @param qm Probability matrix.
#' @param kappa Mixture parameter.
#' @return Mixture matrix.
js_mixture <- function(pm, qm, kappa = 0.5) {
  (kappa * pm) + ((1 - kappa) * qm)
}

klqz_update <- function(inp, out, method) {
  prob_type <- method$prob_type
  if (is.null(prob_type)) {
    stop("Embedding method must have a prob type")
  }
  fn_name <- paste0("klqz_update_p", prob_type)
  fn <- get(fn_name)
  if (is.null(fn)) {
    stop("Unable to find KLQZ update function for ", prob_type)
  }
  fn(inp, out, method)
}

#' Update Output With KL Divergence from Q to Z
klqz_update_pjoint <- function(inp, out, method) {
  out$zm <- js_mixture(inp$pm, out$qm, method$cost$kappa)
  out$kl_qz <- kl_divergence(out$qm, out$zm, method$eps)
  out
}

#' Update Output with KL Divergence from Q to Z per Row
klqz_update_prow <- function(inp, out, method) {
  out$zm <- js_mixture(inp$pm, out$qm, method$cost$kappa)
  out$kl_qz <- kl_divergence_rows(out$qm, out$zm, method$eps)
  out
}

#' JSE fun/grad wrapper
jse_fg <- function(kappa = 0.5) {
  kappa <- clamp(kappa, min_val = sqrt(.Machine$double.eps),
                max_val = 1 - sqrt(.Machine$double.eps))
  list(
    fn = jse_cost_fn,
    gr = jse_cost_gr,
    kappa = kappa,
    kappa_inv = 1 / kappa
  )
}

# JSE cost function wrapper
jse_cost_fn <- function(inp, out, method) {
  jse_cost(inp, out, method)
}
attr(jse_cost_fn, "sneer_cost_type") <- "prob"
attr(jse_cost_fn, "sneer_cost_norm") <- "jse_cost_norm"

# JSE cost gradient
jse_cost_gr <- function(inp, out, method) {
  method$cost$kappa_inv * log((out$qm + method$eps) / (out$zm + method$eps))
}

