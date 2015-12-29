# Functions to generate probabilities from coordinates, as used in
# probability-based embedding methods.

#' Create Weight Matrix from Output Data
#'
#' Creates a matrix of positive weights from the distance matrix of the
#' embedded coordinates.
#'
#' @param out Output data.
#' @param method Embedding method.
#' @return Weight matrix for the embedded coordinates in \code{out}.
weights <- function(out, method) {
  coords_to_weights(out[[method$mat_name]], method$weight_fn)
}

#' Create Weight Matrix from Coordinates Matrix
#'
#' @param ym Matrix of coordinates.
#' @param weight_fn Function with signature \code{weight_fn(d2m)}
#' where \code{d2m} is a matrix of squared distances between the output
#' coordinates. It should return a weight matrix.
#' @return Weight matrix.
coords_to_weights <- function(ym, weight_fn) {
  d2m <- coords_to_dist2(ym)
  dist2_to_weights(d2m, weight_fn)
}

#' Create Weight Matrix from Squared Distances
#'
#' Weights are subsequently normalized to probabilities in probability-based
#' embedding. This function guarantees that the self-weight is 0.
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

#' Create Row Probability Matrix from Weight Matrix
#'
#' Used in ASNE. The probability matrix is such that all elements are positive
#' and each row sums to 1.
#'
#' @param wm Matrix of weighted distances.
#' @return Row probability matrix.
weights_to_prow <- function(wm) {
  row_sums <- apply(wm, 1, sum)
  pm <- sweep(wm, 1, row_sums, "/")
  clamp(pm)
}

#' Create Conditional Probability Matrix from Weight Matrix
#'
#' Used in SSNE and TSNE. The probability matrix is such that all elements are
#' positive and sum to 1.
#'
#' @param wm Matrix of weighted distances.
#' @return Probability matrix.
weights_to_pcond <- function(wm) {
  qm <- wm / sum(wm)
  clamp(qm)
}

#' Create Joint Probability Matrix from Conditional Probability Matrix
#'
#' Used in SSNE and TSNE to convert the input conditional probabilties into a
#' joint probability matrix.
#'
#' @param pm Conditional probability matrix or row stochastic matrix.
#' @return a Joint probability matrix, such that the elements sum to 1 and
#' \code{pm[i, j]} = \code{pm[j, i]}.
pcond_to_pjoint <- function(pm) {
  weights_to_pcond(symmetrize_matrix(pm))
}
