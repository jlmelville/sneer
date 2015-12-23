# Functions defining the SNE-family of similarity embedding methods.

#' t-distributed Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement t-SNE.
#'
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @return a list with the following members:
#' \itemize{
#'  \item \code{cost_fn} Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.
#'  \item \code{weight_fn} Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: t-SNE uses the Student t-distribution with one degree of
#'  freedom.
#'  \item \code{stiffness_fn} Stiffness function.
#'  \item \code{prob_out_fn} Function to convert weights matrix to a specific
#'  probability matrix, e.g. row-based or joint.
#'  \item \code{after_init_fn} Method-specific initialization function to
#'  invoke after input and output initialization callbacks. For this method,
#'  the input probability matrix must be converted from a row probability
#'  matrix to a joint probability matrix.
#' }
#' @references
#' Laurens van der Maarten, Geoffrey Hinton.
#' Visualizing Data using t-SNE.
#' Journal of Machine Learning Research, 2008, 9, 2579-2605.
#' @export
tsne <- function(mat_name = "ym") {
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
    mat_name = mat_name
  )
}

#' Symmetric Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement SSNE.
#'
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @return a list with the following members:
#' \itemize{
#'  \item \code{cost_fn} Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.
#'  \item \code{weight_fn} Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: SSNE uses the exponential function.
#'  \item \code{stiffness_fn} Stiffness function.
#'  \item \code{prob_out_fn} Function to convert weights matrix to a specific
#'  probability matrix, e.g. row-based or joint.
#'  \item \code{after_init_fn} Method-specific initialization function to
#'  invoke after input and output initialization callbacks. For this method,
#'  the input probability matrix must be converted from a row probability
#'  matrix to a joint probability matrix.
#' }
#' @references
#' J.A. Cook, I. Sutskever, A. Mnih, and G.E. Hinton.
#' Visualizing similarity data with a mixture of maps.
#' In Proceedings of the 11th International Conference on Artificial
#' Intelligence and Statistics, volume 2, pages 67-74, 2007.
#'
#' Laurens van der Maarten, Geoffrey Hinton.
#' Visualizing Data using t-SNE.
#' Journal of Machine Learning Research, 2008, 9, 2579-2605.
#' @export
ssne <- function(mat_name = "ym") {
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
    mat_name = mat_name
  )
}

#' Asymmetric Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement ASNE.
#'
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @return a list with the following members:
#' \itemize{
#'  \item \code{cost_fn} Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.
#'  \item \code{weight_fn} Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: ASNE uses the exponential function.
#'  \item \code{stiffness_fn} Stiffness function.
#'  \item \code{prob_out_fn} Function to convert weights matrix to a specific
#'  probability matrix, e.g. row-based or joint.
#'  \item \code{after_init_fn} Method-specific initialization function to
#'  invoke after input and output initialization callbacks.
#' }
#' @references
#' G.E. Hinton and S.T. Roweis.
#' Stochastic Neighbor Embedding.
#' In Advances in Neural Information Processing Systems, volume 15,
#' pages 833-840, Cambridge, MA, USA, 2002. The MIT Press.
#' @export
asne <- function(mat_name = "ym") {
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
    mat_name = mat_name
  )
}

#' t-distributed Asymmetric Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement t-ASNE: a method that
#' I just made up to illustrate how to explore different aspects of embedding
#' within sneer: this uses the t-distributed distance weighting of t-SNE, but
#' for probability generation uses the point-wise distribution of ASNE.
#'
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @return a list with the following members:
#' \itemize{
#'  \item \code{cost_fn} Cost function for the embedding: the Kullback-Leibler
#'  divergence between the input and output probabilities.
#'  \item \code{weight_fn} Weight function for mapping squared distances in the
#'  embedded coordinates to weights, which are in turn converted to
#'  probabilities: tASNE uses the exponential function.
#'  \item \code{stiffness_fn} Stiffness function.
#'  \item \code{prob_out_fn} Function to convert weights matrix to a specific
#'  probability matrix, e.g. row-based or joint.
#'  \item \code{after_init_fn} Method-specific initialization function to
#'  invoke after input and output initialization callbacks.
#' }
#' @export
tasne <- function(mat_name = "ym") {
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
    mat_name = mat_name
  )
}
