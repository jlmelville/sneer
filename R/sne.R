# Functions defining the SNE-family of probability-based embedding methods.

#' Probability-Based Embeddings
#'
#' The available embedding methods which work by reproducing probabilities
#' based on the input distances, rather than the input distances directly.
#'
#' @examples
#' \dontrun{
#' embed_prob(method = tsne(), ...)
#' embed_prob(method = asne(), ...)
#' }
#' @keywords internal
#' @name probability_embedding_methods
#' @family sneer probability embedding methods
NULL

#' t-Distributed Stochastic Neighbor Embedding
#'
#' Creates a list of functions that collectively implement t-SNE.
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#'  \code{wm}.}
#'  \item{\code{wm}}{Weight matrix generated from the distances between points
#'  in \code{ym}.}
#' }
#'
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return a list containing:
#'  \item{\code{cost_fn}}{Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.}
#'  \item{\code{weight_fn}}{Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: t-SNE uses the Student t-distribution with one degree of
#'  freedom.}
#'  \item{\code{stiffness_fn}}{Stiffness function.}
#'  \item{\code{update_out_fn}}{Function to calculate and store any needed
#'  data after a coordinate update.}
#'  \item{\code{after_init_fn}}{Method-specific initialization function to
#'  invoke after input and output initialization callbacks. For this method,
#'  the input probability matrix must be converted from a row probability
#'  matrix to a joint probability matrix.}
#'  \item{\code{eps}}{Small floating point value used to prevent numerical
#'  problems, e.g. in gradients and cost functions.}
#' @references
#' Laurens van der Maarten, Geoffrey Hinton.
#' Visualizing Data using t-SNE.
#' Journal of Machine Learning Research, 2008, 9, 2579-2605.
#' @export
#' @family sneer embedding methods
#' @family sneer probability embedding methods
tsne <- function(eps = .Machine$double.eps) {
  f <- function(pm, qm, wm) {
    4 * (pm - qm) * wm
  }

  list(
    cost_fn = kl_cost,
    weight_fn = tdist_weight,
    stiffness_fn = function(method, inp, out) {
      f(inp$pm, out$qm, out$wm)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_pcond(wm)
      out$wm <- wm
      out
    },
    after_init_fn = function(inp, out, method) {
      inp$pm <- prow_to_pjoint(inp$pm)
      list(inp = inp)
    },
    eps = eps
  )
}

#' Symmetric Stochastic Neighbor Embedding
#'
#' Creates a list of functions that collectively implement SSNE.
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Joint probability matrix based on embedded coordinates.}
#' }
#'
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return a list containing:
#'  \item{\code{cost_fn}}{Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.}
#'  \item{\code{weight_fn}}{Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: SSNE uses the exponential function.}
#'  \item{\code{stiffness_fn}}{Stiffness function.}
#'  \item{\code{update_out_fn}}{Function to calculate and store any needed
#'  data after a coordinate update.}
#'  \item{\code{after_init_fn}}{Method-specific initialization function to
#'  invoke after input and output initialization callbacks. For this method,
#'  the input probability matrix must be converted from a row probability
#'  matrix to a joint probability matrix.}
#'  \item{\code{eps}}{Small floating point value used to prevent numerical
#'  problems, e.g. in gradients and cost functions.}
#' @references
#' J.A. Cook, I. Sutskever, A. Mnih, and G.E. Hinton.
#' Visualizing similarity data with a mixture of maps.
#' In Proceedings of the 11th International Conference on Artificial
#' Intelligence and Statistics, volume 2, pages 67-74, 2007.
#'
#' Laurens van der Maarten, Geoffrey Hinton.
#' Visualizing Data using t-SNE.
#' Journal of Machine Learning Research, 2008, 9, 2579-2605.
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @export
ssne <- function(eps = .Machine$double.eps) {
  f <- function(pm, qm) {
    4 * (pm - qm)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(method, inp, out) {
      f(inp$pm, out$qm)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_pcond(wm)
      out
    },
    after_init_fn = function(inp, out, method) {
      inp$pm <- prow_to_pjoint(inp$pm)
      list(inp = inp)
    },
    eps = .Machine$double.eps
  )
}

#' Asymmetric Stochastic Neighbor Embedding
#'
#' Creates a list of functions that collectively implement ASNE.
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Row probability matrix based on embedded coordinates.}
#' }
#'
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return a list containing:
#'  \item{\code{cost_fn}}{Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.}
#'  \item{\code{weight_fn}}{Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: ASNE uses the exponential function.}
#'  \item{\code{stiffness_fn}}{Stiffness function.}
#'  \item{\code{update_out_fn}}{Function to calculate and store any needed
#'  data after a coordinate update.}
#'  \item{\code{after_init_fn}}{Method-specific initialization function to
#'  invoke after input and output initialization callbacks.}
#'  \item{\code{eps}}{Small floating point value used to prevent numerical
#'  problems, e.g. in gradients and cost functions.}
#' @references
#' G.E. Hinton and S.T. Roweis.
#' Stochastic Neighbor Embedding.
#' In Advances in Neural Information Processing Systems, volume 15,
#' pages 833-840, Cambridge, MA, USA, 2002. The MIT Press.
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @export
asne <- function(eps = .Machine$double.eps) {
  f <- function(pm, qm) {
    km <- 2 * (pm - qm)
    km + t(km)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(method, inp, out) {
      f(inp$pm, out$qm)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_prow(wm)
      out
    },
    after_init_fn = function(inp, out, method) {
      inp$pm <- clamp(inp$pm)
      list(inp = inp)
    },
    eps = eps
  )
}

#' t-distributed Asymmetric Stochastic Neighbor Embedding
#'
#' Creates a list of functions that collectively implement t-ASNE: a method that
#' I just made up to illustrate how to explore different aspects of embedding
#' within sneer: this uses the t-distributed distance weighting of t-SNE, but
#' for probability generation uses the point-wise distribution of ASNE.
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates.}
#'  \item{\code{qm}}{Row probability matrix based on the weight matrix
#'  \code{wm}.}
#'  \item{\code{wm}}{Weight matrix generated from the distances between points
#'  in \code{ym}.}
#' }
#'
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return a list containing:
#'  \item{\code{cost_fn}}{Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.}
#'  \item{\code{weight_fn}}{Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: t-ASNE uses the exponential function.}
#'  \item{\code{stiffness_fn}}{Stiffness function.}
#'  \item{\code{update_out_fn}}{Function to calculate and store any needed
#'  data after a coordinate update.}
#'  \item{\code{after_init_fn}}{Method-specific initialization function to
#'  invoke after input and output initialization callbacks.}
#'  \item{\code{eps}}{Small floating point value used to prevent numerical
#'  problems, e.g. in gradients and cost functions.}
#' @family sneer embedding methods
#' @family sneer probability embedding methods
#' @export
tasne <- function(eps = .Machine$double.eps) {
  f <- function(pm, qm, wm) {
    km <- 2 * (pm - qm) * wm
    km + t(km)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = tdist_weight,
    stiffness_fn = function(method, inp, out) {
      f(inp$pm, out$qm, out$wm)
    },
    update_out_fn = function(inp, out, method) {
      wm <- weights(out, method)
      out$qm <- weights_to_prow(wm)
      out$wm <- wm
      out
    },
    after_init_fn = function(inp, out, method) {
      inp$pm <- clamp(inp$pm)
      list(inp = inp)
    },
    eps = eps
  )
}
