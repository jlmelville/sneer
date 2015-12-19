# Functions defining the SNE-family of similarity embedding methods.

#' t-distributed Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement t-SNE.
#'
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
tsne <- function() {
  f <- function(pm, qm, wm) {
    4 * (pm - qm) * wm
  }

  list(
    cost_fn = kl_cost,
    weight_fn = tdist_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm, out$wm)
    },
    prob_out_fn = weights_to_pcond,
    update_out_fn = function(inp, out, stiffness, wm) {
      out$wm <- wm
      out
    },
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- prow_to_pjoint(inp$pm)
      list(inp = inp)
    }
  )
}

#' Symmetric Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement SSNE.
#'
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
ssne <- function() {
  f <- function(pm, qm) {
    4 * (pm - qm)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm)
    },
    prob_out_fn = weights_to_pcond,
    update_out_fn = NULL,
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- prow_to_pjoint(inp$pm)
      list(inp = inp)
    }
  )
}

#' Asymmetric Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement ASNE.
#'
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
asne <- function() {
  f <- function(pm, qm) {
    km <- 2 * (pm - qm)
    km + t(km)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm)
    },
    prob_out_fn = weights_to_prow,
    update_out_fn = NULL,
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- clamp(inp$pm)
      list(inp = inp)
    }
  )
}

#' t-distributed Asymmetric Stochastic Neighbor Embedding.
#'
#' Creates a list of functions that collectively implement t-ASNE: a method that
#' I just made up to illustrate how to explore different aspects of embedding
#' within sneer: this uses the t-distributed distance weighting of t-SNE, but
#' for probability generation uses the point-wise distribution of ASNE.
#'
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
tasne <- function() {
  f <- function(pm, qm, wm) {
    km <- 2 * (pm - qm) * wm
    km + t(km)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = tdist_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm, out$wm)
    },
    prob_out_fn = weights_to_prow,
    update_out_fn = function(inp, out, stiffness, wm) {
      out$wm <- wm
      out
    },
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- clamp(inp$pm)
      list(inp = inp)
    }
  )
}
