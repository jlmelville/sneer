# Cost functions. Used by optimization routines to improve the embedding.

#' Calculate the Kullback Leibler divergence from input and output data.
#'
#' @param inp Input data.
#' @param out Output data.
#' @return the KL divergence between \code{inp$pm} and \code{out$qm}.
kl_cost <- function(inp, out) {
  kl_divergence(inp$pm, out$qm)
}
attr(kl_cost, "sneer_cost_type") <- "prob"

#' Calculate the Kullback Leibler divergence between two matrices of
#' probabilities.
#'
#' If the matrices represent row probabilities, then the returned divergence
#' is the average over the divergence for each row.
#'
#' @param pm Probability Matrix. First probability in the divergence.
#' @param qm Probability Matrix. Second probability in the divergence.
#' @return the KL divergence between \code{pm} and \code{qm}.
kl_divergence <- function(pm, qm) {
  sum(apply(pm * log((pm + .Machine$double.eps) /
                       (qm + .Machine$double.eps)), 1, sum)) / sum(pm)
}

#' Factory function to normalize a cost function.
#'
#' The cost function can be any function where the more positive the value,
#' the worse the solution is considered to be. The corresponding normalized
#' version is that which scales the cost so that a "null" model would give a
#' normalized cost of 1.0.
#'
#' The definition of a "null" model is one which is as good as can be if one
#' didn't use any information from the data at all. For methods that attempt
#' to preserve distances, this would be equivalent to making all the embedded
#' distances the same, which can only be achieved by making them all zero. For
#' similarity-preserving methods, the equivalent would be to make all the
#' probabilities equal.
#'
#' It might also be possible to compare embeddings between different methods,
#' but it's safer to simply use it with one method, and know that while an
#' embedding with a normalized cost of e.g. 0.85 is poor, one with a normalized
#' cost of 1.2 is basically worse than guessing.
#'
#' @param cost_fn Cost function. Should have the signature
#' \code{cost_fn(inp, out)} and return a scalar numeric cost value. In addition
#' it should have an appropriate \code{sneer_cost_type} attribute set. For cost
#' functions that act on probabilities, this should be \code{"prob"}. For cost
#' function that act on distances, this should be \code{"dist"}.
#' @return Normalized cost function with the signature
#' \code{norm_fn(inp, out)} and which return a scalar numeric cost value.
make_normalized_cost_fn <- function(cost_fn) {
  cost_type <- attr(cost_fn, "sneer_cost_type")
  null_model_fn_name <- paste("null_model", cost_type, sep = "_")
  if (cost_type == "prob") {
    mat_name <- "qm"
  }
  else if (cost_type == "dist") {
    mat_name <- "dm"
  }
  else {
    stop("No known null model matrix name for cost type '", cost_type, "'")
  }
  function(inp, out) {
    cost <- cost_fn(inp, out)
    out[[mat_name]] <- do.call(null_model_fn_name, list(out[[mat_name]]))
    null_cost <- cost_fn(inp, out)
    cost / null_cost
  }
}

#' For a given probability matrix, return the equivalent "null" model, i.e. one
#' where all probabilities are equal.
#'
#' @param pm Probability matrix. Can be row, joint or conditional.
#' @return Probability matrix where all elements are equal.
null_model_prob <- function(pm) {
  matrix(sum(pm) / (nrow(pm) * ncol(pm)), nrow = nrow(pm), ncol = ncol(pm))
}
