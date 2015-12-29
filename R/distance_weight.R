# Distance weighting functions. Used to weight distances before being converted
# probabilities in probability-based embedding. These functions work on the
# squared distances.

#' Exponential Weight
#'
#' Applies exponential weighting to the squared distances.
#'
#' The weight matrix, \eqn{W} is generated from the squared distances matrix,
#' \eqn{D2} by:
#'
#' \deqn{W = e^{-\beta D2}}{exp(-beta * D2)}
#'
#' @param d2m Matrix of squared distances.
#' @param beta exponential parameter.
#' @return Weight matrix.
#' @family sneer weight functions
#' @export
exp_weight <- function(d2m, beta = 1) {
  exp(-beta * d2m)
}

#' Exponential Distance Weight
#'
#' Applies exponential weighting to the distances, rather than the square of
#' the distances.
#'
#' Included so results can be compared with other implementations
#' of TSNE, although you'd normally use the squared distances. The weight
#' matrix, \eqn{W} is generated from the squared distances matrix,
#' \eqn{D2} by:
#'
#' \deqn{W = e^{-\beta \sqrt{D2}}}{W = exp(-beta * sqrt(D2))}
#'
#' @param d2m Matrix of squared distances.
#' @param beta Exponential parameter.
#' @return Weight matrix.
#' @family sneer weight functions
#' @export
sqrt_exp_weight <- function(d2m, beta = 1) {
  exp(-beta * sqrt(d2m))
}

#' Student-t Distribution Weight
#'
#' Applies weighting using the Student-t distribution with one degree of
#' freedom.
#'
#' Compared to the exponential weighting this has a much heavier tail.
#' The weight matrix, \eqn{W} is generated from the squared distances matrix,
#' \eqn{D2} by:
#' \deqn{W = \frac{1}{(1 + D2)}}{W = 1/(1 + D2)}
#'
#' @param d2m Matrix of squared distances.
#' @return Weight matrix.
#' @family sneer weight functions
#' @export
tdist_weight <- function(d2m) {
  1 / (1 + d2m)
}
