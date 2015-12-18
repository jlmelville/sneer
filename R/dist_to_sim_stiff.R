# Functions to generate probabilities from coordinates, as used in similarity
# based embedding methods.

#' Generate a weights matrix from squared coordinates.
#'
#' Weights are subsequently normalized to probabilities in similarity embedding.
#' This function guarantees that the self-weight is 0.
#'
#' @param d2m Matrix of squared distances.
#' @param weight_fn Function with signature \code{weight_fn(d2m)} where
#' \code{d2m} is a matrix of squared distances.
#' @return Weights matrix. The diagonal (i.e. self-weight) is enforced to be
#' zero.
dist2_to_weights <- function(d2m, weight_fn) {
  wm <- weight_fn(d2m)
  diag(wm) <- 0  # set self-weights to 0
  wm
}

#' Update matrices that are dependent on embedding coordinates.
#'
#' After embedding coordinates are updated, other matrices may need
#' to be also updated: e.g. distance matrix and for similarity-based
#' embedding weights and probability matrices.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param stiffness Stiffness configuration.
#' @param mat_name Name of the matrix in \code{out} which contains the
#' coordinates.
#' @return Updated version of \code{out}.
update_out <- function(inp, out, stiffness, mat_name = "ym") {
  d2m <- coords_to_dist2(out[[mat_name]])

  wm <- dist2_to_weights(d2m, stiffness$weight_fn)
  out$qm <- stiffness$prob_out_fn(wm)

  if (!is.null(stiffness$update_out_fn)) {
    out <- stiffness$update_out_fn(inp, out, stiffness, wm)
  }

  out
}


#' Converts a weight matrix to a row probability matrix.
#'
#' Used in ASNE. The probability matrix is such that all elements are positive
#' and each row sums to 1.
#'
#' @param wm a matrix of weighted distances.
#' @return a row probability matrix.
weights_to_prow <- function(wm) {
  row_sums <- apply(wm, 1, sum)
  pm <- sweep(wm, 1, row_sums, "/")
  clamp(pm)
}

#' Converts a weights matrix to a probability matrix.
#'
#' Used in SSNE and TSNE. The probability matrix is such that all elements are
#' positive and sum to 1.
#'
#' @param wm a matrix of weighted distances.
#' @return The probability matrix.
weights_to_pcond <- function(wm) {
  qm <- wm / sum(wm)
  clamp(qm)
}

#' Convert a conditional probability matrix to a joint probability matrix.
#'
#' Used in SSNE and TSNE to convert the input conditional probabilties into a
#' joint probability matrix.
#'
#' @param pm a conditional probability matrix or row stochastic matrix.
#' @return a joint probability matrix, such that the elements sum to 1 and
#' \code{pm[i, j]} = \code{pm[j, i]}.
pcond_to_pjoint <- function(pm) {
  weights_to_pcond(symmetrize_matrix(pm))
}
