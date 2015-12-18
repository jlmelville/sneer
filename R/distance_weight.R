# Distance weighting functions. Used to weight distances before being converted
# probabilities in similarity-based embedding. These functions work on the
# squared distances.

#' Exponential weight.
#'
#' Applies exponential weighting to the squared distances. Can also be thought
#' of as gaussian weighting of the distances.
#'
#' \eqn{W = e^{-\beta D^2}}
#'
#' @param d2m Matrix of squared distances.
#' @param beta xponential parameter.
#' @return Weight matrix.
exp_weight <- function(d2m, beta = 1) {
  exp(-beta * d2m)
}

#' Exponential weighting of the distances.
#'
#' Applies exponential weighting to the distances, rather than the square of
#' the distances. Included so results can be compared with other implementations
#' of TSNE, although you'd normally use the squared distances.
#'
#' \eqn{W = e^{-\beta D}}
#'
#' @param d2m Matrix of squared distances.
#' @param beta Exponential parameter.
#' @return Weight matrix.
sqrt_exp_weight <- function(d2m, beta = 1) {
  exp(-beta * sqrt(d2m))
}

#' Student-t distribution weighting.
#'
#' Applies weighting using the Student-t distribution with one degree of
#' freedom. Compared to the exponential weighting this has a much heavier tail.
#'
#' \eqn{W = \frac{1}{(1 + D^2)}}
#'
#' @param d2m Matrix of squared distances.
#' @return Weight matrix.
tdist_weight <- function(d2m) {
  1 / (1 + d2m)
}
