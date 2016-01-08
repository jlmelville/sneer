#' Neighbor Retrieval Visualizer (NeRV)
#'
#' A probability-based embedding method.
#'
#' NeRV is a variant of Asymmetric Stochastic Neighbor Embedding
#' (see \code{\link{asne}}), with a modified cost function: in addition to
#' calculating the Kullback Leibler divergence of the output probabilities Q,
#' from the input probabilities, P, it also includes the divergence of P from Q.
#' The final cost function is a weighted sum of these two individual functions.
#' Hence ASNE is a special case of NeRV where all the weight is placed on the
#' first component of the cost function.
#'
#' From an information retrieval perspective, the weighting factor allows the
#' user to place a relative weight on false positives: points on the embedded
#' map which have a close distance, but a low input probability, i.e. should not
#' have been embedded as close neighbors, versus false negatives: pairs with a
#' large distance in the output coordinates, but a high input probability, i.e.
#' should have been embedded as close neighbors. From this perspective, ASNE
#' is the equivalent of emphasising false positives over false negatives.
#'
#' The probability matrix used in NeRV:
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
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to ASNE. Must be a value between 0 and 1.
#' @param eps Small floating point value used to prevent numerical problems,
#'   e.g. in gradients and cost functions.
#' @param verbose If \code{TRUE}, log information about the embedding.
#' @return An embedding method for use by an embedding function.
#' @references
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#' @seealso NeRV uses the \code{\link{nerv_cost}} cost function and the
#'   \code{\link{exp_weight}} similarity function for converting distances to
#'   probabilities.
#' The return value of this function should be used with the
#' \code{\link{embed_prob}} embedding function.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @examples
#' \dontrun{
#' # default NeRV settings
#' embed_prob(method = nerv(lambda = 0.5), ...)
#'
#' # equivalent to ASNE
#' embed_prob(method = nerv(lambda = 1), ...)
#'
#' # puts an emphasis on only keeping true neighbors close together
#' # tends to produce a larger number of small, tight clusters
#' embed_prob(method = nerv(lambda = 0), ...)
#' }
nerv <- function(lambda = 0.5, eps = .Machine$double.eps, verbose = TRUE) {
  list(
    cost_fn = nerv_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(method, inp, out) {
      nerv_stiffness(inp$pm, out$qm, out$rev_kl, lambda, eps)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_prow(wm)
      out$rev_kl <- reverse_kl_cost(inp, out, method)
      out
    },
    prob_type = "row",
    eps = eps,
    lambda = lambda
  )
}


#' t-Distributed Neighbor Retrieval Visualizer (t-NeRV)
#'
#' A probability-based embedding method.
#'
#' t-NeRV is a variant of t-distributed Stochastic Neighbor Embedding
#' (\code{\link{tsne}}), with a modified cost function: in addition to
#' calculating the Kullback Leibler divergence of the output probabilities Q,
#' from the input probabilities, P, it also includes the divergence of P from Q.
#' The final cost function is a weighted sum of these two individual functions.
#' Hence SSNE is a special case of NeRV where all the weight is placed on the
#' first component of the cost function.
#'
#' From an information retrieval perspective, the weighting factor allows the
#' user to place a relative weight on false positives: points on the embedded
#' map which have a close distance, but a low input probability, i.e. should not
#' have been embedded as close neighbors, versus false negatives: pairs with a
#' large distance in the output coordinates, but a high input probability, i.e.
#' should have been embedded as close neighbors. From this perspective, t-SNE
#' is the equivalent of emphasising false positives over false negatives.
#'
#' The probability matrix used in t-NeRV:
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
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
#' @param eps Small floating point value used to prevent numerical problems,
#'   e.g. in gradients and cost functions.
#' @param verbose If \code{TRUE}, log information about the embedding.
#' @return An embedding method for use by an embedding function.
#' @references
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#' @seealso NeRV uses the \code{\link{nerv_cost}} cost function and the
#'   \code{\link{tdist_weight}} similarity function for converting distances to
#'   probabilities.
#' The return value of this function should be used with the
#'  \code{\link{embed_prob}} embedding function.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @examples
#' \dontrun{
#' # default t-NeRV settings
#' embed_prob(method = tnerv(lambda = 0.5), ...)
#'
#' # equivalent to t-SNE
#' embed_prob(method = tnerv(lambda = 1), ...)
#'
#' # puts an emphasis on only keeping true neighbors close together
#' # tends to produce a larger number of small, tight clusters
#' embed_prob(method = tnerv(lambda = 0), ...)
#' }
tnerv <- function(eps = .Machine$double.eps, lambda = 0.5, verbose = TRUE) {
  list(
    cost_fn = nerv_cost,
    weight_fn = tdist_weight,
    stiffness_fn = function(method, inp, out) {
      tnerv_stiffness(inp$pm, out$qm, out$wm, out$rev_kl, lambda, eps)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_pcond(wm)
      out$rev_kl <- reverse_kl_cost(inp, out, method)
      out$wm <- wm
      out
    },
    prob_type = "joint",
    eps = eps,
    lambda = lambda
  )
}

#' Symmetric Neighbor Retrieval Visualizer (SNeRV)
#'
#' A probability-based embedding method.
#'
#' SNeRV is a "symmetric" variant of \code{\link{nerv}}. Rather than use the
#' conditional point-based probabilities of \code{\link{asne}}, it uses the
#' joint pair-based probabilities of \code{\link{ssne}}. Empirically, this seems
#' to provides better convergence behavior when the \code{lambda} parameter is
#' set to small values. When \code{lambda = 1}, this method is equivalent to
#' SSNE.
#'
#' The probability matrix used in SNeRV:
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
#'  \item{\code{qm}}{Joint probability matrix.}
#' }
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
#' @param eps Small floating point value used to prevent numerical problems,
#'   e.g. in gradients and cost functions.
#' @param verbose If \code{TRUE}, log information about the embedding.
#' @return An embedding method for use by an embedding function.
#' @references
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#'
#' Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
#' Heavy-tailed symmetric stochastic neighbor embedding.
#' In \emph{Advances in neural information processing systems} (pp. 2169-2177).
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#' @seealso SNeRV uses the \code{\link{nerv_cost}} cost function and the
#'   \code{\link{exp_weight}} similarity function for converting distances to
#'   probabilities.
#' The return value of this function should be used with the
#'   \code{\link{embed_prob}} embedding function.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @examples
#' \dontrun{
#' # default SNeRV settings
#' embed_prob(method = snerv(lambda = 0.5), ...)
#'
#' # equivalent to SSNE
#' embed_prob(method = snerv(lambda = 1), ...)
#'
#' # puts an emphasis on only keeping true neighbors close together
#' # tends to produce a larger number of small, tight clusters
#' embed_prob(method = snerv(lambda = 0), ...)
#' }
snerv <- function(eps = .Machine$double.eps, lambda = 0.5, verbose = TRUE) {
  list(
    cost_fn = nerv_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(method, inp, out) {
      snerv_stiffness(inp$pm, out$qm, out$wm, out$rev_kl, lambda, eps)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_pcond(wm)
      out$rev_kl <- reverse_kl_cost(inp, out, method)
      out$wm <- wm
      out
    },
    prob_type = "joint",
    eps = eps,
    lambda = lambda
  )
}

#' Heavy-tailed Symmetric Neighbor Retrieval Visualizer (HSNeRV)
#'
#' A probability-based embedding method.
#'
#' HSNeRV is a hybrid of \code{\link{snerv}} and \code{\link{hssne}}. It has
#' the \code{lambda} parameter of SNeRV, allowing for the control of precision
#' versus recall, and the \code{alpha} and \code{beta} parameters of HSSNE which
#' give the behavior of SSNE/SNeRV when \code{alpha} is close to zero,
#' and of t-SNE/t-NeRV when \code{alpha = 1}.
#'
#' The probability matrix used in HSNeRV:
#'
#' \itemize{
#'  \item{represents one probability distribution, i.e. the grand sum of the
#'  matrix is one.}
#'  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#'  probabilities are joint probabilities.}
#' }
#'
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#'  \code{wm}.}
#' }
#' @param lambda Weighting factor controlling the emphasis placed on precision
#'   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#'   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
#' @param alpha Tail heaviness. Must be greater than zero. When set to a small
#'   value this method is equivalent to SSNE or SNeRV (depending on the value
#'   of \code{lambda}. When set to one to one, this method behaves like
#'   t-SNE/t-NeRV.
#' @param beta The precision of the function. Becomes equivalent to the
#'   precision in the Gaussian distribution of distances as \code{alpha}
#'   approaches zero.
#' @param eps Small floating point value used to prevent numerical problems,
#'   e.g. in gradients and cost functions.
#' @param verbose If \code{TRUE}, log information about the embedding.
#' @return An embedding method for use by an embedding function.
#' @references
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#' @seealso HSNeRV uses the \code{\link{nerv_cost}} cost function and the
#'   \code{\link{heavy_tail_weight}} similarity function for converting
#'   distances to probabilities.
#'
#' The return value of this function should be used with the
#'   \code{\link{embed_prob}} embedding function.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @examples
#' \dontrun{
#' # equivalent to SSNE
#' embed_prob(method = hsnerv(lambda = 1, alpha = 1.5e-8), ...)
#'
#' # equivalent to t-SNE
#' embed_prob(method = hsnerv(lambda = 1, alpha = 1), ...)
#'
#' # equivalent to default SNeRV
#' embed_prob(method = hsnerv(lambda = 0.5, alpha = 1.5e-8), ...)
#'
#' # equivalent to default t-NeRV
#' embed_prob(method = hsnerv(lambda = 0.5, alpha = 1), ...)
#'
#' }
hsnerv <- function(lambda = 0.5, alpha = 1.5e-8, beta = 1,
                   eps = .Machine$double.eps, verbose = TRUE) {
  weight_fn <- function(D2) {
    heavy_tail_weight(D2, beta, alpha)
  }
  attr(weight_fn, "type") <- attr(heavy_tail_weight, "type")


  list(
    cost_fn = nerv_cost,
    weight_fn = weight_fn,
    stiffness_fn = function(method, inp, out) {
      hsnerv_stiffness(inp$pm, out$qm, out$wm, out$rev_kl, lambda, alpha, beta,
                       eps)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_pcond(wm)
      out$rev_kl <- reverse_kl_cost(inp, out, method)
      out$wm <- wm
      out
    },
    prob_type = "joint",
    eps = eps,
    lambda = lambda
  )
}

#' Neighbor Retrieval Visualizer (NeRV) Cost Function
#'
#' A measure of embedding quality between input and output data.
#'
#' This cost function evaluates the embedding quality by calculating a weighted
#' sum of two KL divergence calculations:
#'
#' \deqn{C_{NeRV} = \lambda D_{KL}(P||Q) + (1-\lambda)D_{KL}(Q||P)}{C_NeRV = [lambda * KL(P||Q)] + [(1-lambda) * KL(Q||P)]}
#'
#' where P is the input probability matrix, Q the output probability matrix, and
#' \eqn{\lambda}{lambda} is a weighting factor between zero and one.
#'
#' This cost function requires the following matrices and values to be defined:
#' \describe{
#'  \item{\code{inp$pm}}{Input probabilities.}
#'  \item{\code{out$qm}}{Output probabilities.}
#'  \item{\code{method$lambda}}{Weighting factor between 0 and 1.}
#' }
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return NeRV cost.
#' @references
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#' @family sneer cost functions
#' @export
nerv_cost <- function(inp, out, method) {
  method$lambda * kl_cost(inp, out, method) +
    (1 - method$lambda) * reverse_kl_cost(inp, out, method)
}
attr(nerv_cost, "sneer_cost_type") <- "prob"


#' Reverse Kullback Leibler Divergence Cost Function
#'
#' A measure of embedding quality between input and output data.
#'
#' This cost function the embedding quality by calculating the KL divergence
#' between the input probabilities and the output probabilities, where the
#' output probabilities are considered the reference probabilities.
#'
#' This cost function requires the following matrices to be defined:
#' \describe{
#'  \item{\code{inp$pm}}{Input probabilities.}
#'  \item{\code{out$qm}}{Output probabilities.}
#' }
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return KL divergence between \code{inp$pm} and \code{out$qm}.
#' @seealso \code{\link{kl_cost}} provides more detail on the differences
#'   between the usual KL divergence and this "reverse" divergence.
#' @family sneer cost functions
#' @export
reverse_kl_cost <- function(inp, out, method) {
  kl_divergence(out$qm, inp$pm, method$eps)
}
attr(kl_cost, "sneer_cost_type") <- "prob"
