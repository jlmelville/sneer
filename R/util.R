# Miscellaneous small functions.

#' Partially apply a function.
#'
#' @param f Function to partially apply.
#' @param ... params of \code{f} to apply.
#' @return Partially applied version of \code{f}.
#' @export
partial <- function(f, ...) {
  args <- list(...)
  function(...) {
    do.call(f, c(args, list(...)))
  }
}

#' Clamp numerical values.
#'
#' Values are truncated so that they lie within (\code{min_val, max_val}). In
#' embedding this is used to prevent individual probabilities values getting
#' too small and causing underflow or some other horrible explosion.
#'
#' @param x Matrix.
#' @param min_val Minimum value allowed for any element in the matrix.
#' @param max_val Maximum value allowed for any element in the matrix.
#' @return Matrix with the clamped values.
clamp <- function(x, min_val = .Machine$double.eps, max_val = NULL) {
  x[x < min_val] <- min_val
  if (!is.null(max_val)) {
    x[x > max_val] <- max_val
  }
  x
}

#' Length of a vector (or matrix)
#'
#' @param x Matrix.
#' @return Length (2-norm) of the matrix.
length_vec <- function(x) {
  sqrt(sum(x ^ 2))
}

#' Scale a vector (or matrix) to length 1.
#'
#' @param x Matrix.
#' @return \code{x} with elements scaled such that its length equals 1.
normalize <- function(x) {
  x / length_vec(x)
}

#' Relative tolerance.
#'
#' @param x real value.
#' @param y real value.
#' @return the relative tolerance between the two values.
reltol <- function(x, y) {
  abs(x - y) / min(abs(x), abs(y))
}

#' Summary of distribution of data.
#'
#' @param vals Array or matrix of data.
#' @param msg Label to identify the data summary.
summarize <- function(vals, msg = "") {
  if (class(vals) == "matrix") {
    vals <- array(vals)
  }
  message(msg, ": ", paste(names(summary(vals)), ":", summary(vals), "|",
                           collapse = ""))
}

#' Euclidean Distance matrix.
#'
#' Creates an Euclidean distance matrix with the type "\code{matrix}", rather
#' than an  object of class "\code{dist}", which the \code{stats} function
#' \code{dist} produces.
#'
#' @param xm Matrix of coordinates.
#' @param min_dist Truncate any inter-point distances in \code{xm} less than
#' this value.
#' @return Distance matrix.
distance_matrix <- function(xm, min_dist = .Machine$double.eps) {
  if (class(xm) != "dist") {
    dm <- as.matrix(dist(xm))
  } else {
    dm <- as.matrix(xm)
  }
  dm <- clamp(dm, min_dist)
}

#' Upper triangle of a matrix as a vector.
#'
#' Useful if you need to get all of the distances in a distance matrix, without
#' including self-distances or double counting. However, all index information
#' is lost.
#'
#' @param x Matrix.
#' @return Vector of elements in the upper triangle of the matrix.
upper_tri <- function(x) {
  x[upper.tri(x)]
}
