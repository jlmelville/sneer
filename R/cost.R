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

#' Factory function to convert a cost function to a stress function.
#'
#' The cost function can be any function where the more positive the value,
#' the worse the solution is considered to be. The corresponding stress
#' function is that which scales the cost so that a "null" model would give a
#' stress of 1.0.
#'
#' This definition of stress is in analogy with the Kruskal stress used in
#' classical MDS. A "null" model is considered to be one where e.g. for
#' distance-based methods the distances are equal (i.e. zero) or, for similarity
#' methods, where all probabilities are equal. The point is to scale the
#' cost so that embeddings that do worse than this are obvious, no matter
#' what the divergence or weighting functions are used.
#'
#' It might also be possible to compare embeddings between different methods,
#' but it's safer to simply use it with one method, and know that while an
#' embedding with a stress of e.g. 0.85 is poor, one with a stress of 1.2 is
#' worse than random.
#'
#' @param cost_fn Cost function. Should have the signature
#' \code{cost_fn(inp, out)} and return a scalar numeric cost value. In addition
#' it should have an appropriate \code{sneer_cost_type} attribute set. For cost
#' functions that act on probabilities, this should be \code{"prob"}. For cost
#' function that act on distances, this should be \code{"dist"}.
#' @param mat_name The name of the matrix in the \code{out} list that will be
#' used by the \code{cost_fn} to calculate the cost.
#' @return Stress function with the signature \code{stress_fn(inp, out)} and
#' which return a scalar numeric stress value.
make_stress_fn <- function(cost_fn, mat_name = "qm") {
  cost_type <- attr(cost_fn, "sneer_cost_type")
  null_model_fn_name <- paste("null_model", cost_type, sep = "_")

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
