# Code to implement w-SSNE as described by Yang and co-workers.

#' Degree Centrality
#'
#' Calculates the degree centrality of a probability matrix.
#'
#' The probability matrix is interpreted as an adjacency matrix of a fully
#' connected graph, with the probability pij being the weight of the edge
#' connecting node i and j. The degree centrality is then the sum of the
#' edges incident to that node. The paper by Yang and co-workers only considers
#' the joint probability matrix used by SSNE and related methods: in order to
#' account for ASNE, where pij != pji, and hence is effectively a directed
#' graph, the degree centrality is scaled by averaging over the indegree
#' centrality and outdegree centrality and then normalizing over the sum of the
#' matrix. This leaves the joint probability results unchanged, and makes the
#' row or conditional probability values identical to the joint probabilities.
#'
#' @param inp Input data
#' @param out Output data
#' @param method Embedding method
#' @return list containing the updated method.
#'
#' @references
#' Yang, Z., Peltonen, J., & Kaski, S. (2014).
#' Optimization equivalence of divergences improves neighbor embedding.
#' In \emph{Proceedings of the 31st International Conference on Machine Learning (ICML-14)}
#' (pp. 460-468).
degree_centrality <- function(inp, out, method) {
  deg <- (colSums(inp$pm) + rowSums(inp$pm)) / (2 * sum(inp$pm))
  if (method$verbose) {
    summarize(deg, "degC")
  }
  method$kernel$im <- outer(deg, deg)
  list(method = method)
}

#' Convert an Embedding Method to a Weighted Version
#'
#' Modifies the similarity kernel of an embedding method so that each
#' weight is further multiplied by the importance of the two observations the
#' weight was based on.
#'
#' The new weight matrix, which we'll call the "importance" weight matrix, is
#' calculated as
#'
#' \deqn{W_{imp} = WM}{W_imp = W*M}
#'
#' where W is the original weight matrix and M is the importance matrix, which
#' is the outer product of an importance vector d with the same length as the
#' number of observations in the dataset, i.e.:
#'
#' \deqn{M_{ij} = d_{i}d_{j}}{M_ij = di * dj}
#'
#' In the originating paper by Yang and co-workers, they suggest using the
#' degree centrality of each observation as the importance. This involves
#' interpreting the input probability matrix as a weighted adjacency matrix,
#' i.e. treat the dataset as a fully connected graph, with weighted edges, where
#' the weight for the edge between observation i and j is the input probability,
#' p_ij.
#'
#' @param method Embedding method to convert into an importance weighted
#' version.
#' @return Converted embedding method.
#'
#' @references
#' Yang, Z., Peltonen, J., & Kaski, S. (2014).
#' Optimization equivalence of divergences improves neighbor embedding.
#' In \emph{Proceedings of the 31st International Conference on Machine Learning (ICML-14)}
#' (pp. 460-468).
importance_weight <- function(method) {
  method$kernel <- imp_kernel(method$kernel)
  method$inp_updated_fn <- degree_centrality
  method
}

#' Convert an existing similarity kernel to an importance weighted version.
#'
#' The original kernel is used to produce the weight matrix (or the derivative)
#' as usual, then multipled by the importance matrix. This should be stored
#' as \code{kernel$im}.
#'
#' @param kernel Similarity kernel to convert.
#' @return Kernel with importance weighting.
imp_kernel <- function(kernel) {
  kernel$orig_fn <- kernel$fn
  kernel$orig_gr <- kernel$gr
  kernel$fn <- function(kernel, d2m) {
    kernel$orig_fn(kernel, d2m) * kernel$im
  }
  attr(kernel$fn, "type") <- attr(kernel$orig_fn, "type")
  kernel$gr <- function(kernel, d2m) {
    kernel$orig_gr(kernel, d2m) * kernel$im
  }

  kernel
}


