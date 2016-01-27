# Miscellaneous small functions.

#' Partial Function Application
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

#' Clamp Numerical Values
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

#' Clamp Scalar Numerical Value
#'
#' Value is truncated so that it lies within (\code{min, max}).
#'
#' @param x Value.
#' @param min If \code{x} is smaller than this value, it will be truncated to
#'  this value.
#' @param max If \code{x} is larger than this value, it will be truncated to
#'  this value.
#' @return Clamped value.
sclamp <- function(x, min, max) {
  base::min(base::max(x, min), max)
}

#' Length of a Vector (or Matrix)
#'
#' @param x Matrix.
#' @return Length (2-norm) of the matrix.
length_vec <- function(x) {
  sqrt(sum(x ^ 2))
}

#' Scale a Vector (or Matrix) to Length 1
#'
#' @param x Matrix.
#' @return \code{x} with elements scaled such that its length equals 1.
normalize <- function(x) {
  x / length_vec(x)
}

#' Relative Tolerance
#'
#' @param x real value.
#' @param y real value.
#' @return the relative tolerance between the two values.
reltol <- function(x, y) {
  abs(x - y) / min(abs(x), abs(y))
}

#' Summarise Data Distribution
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

#' Euclidean Distance Matrix
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

#' Upper Triangle of a Matrix as a Vector
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

#' Remove NULL Members from a List
#'
#' @param l List.
#' @return List with NULL members removed.
#' @examples
#' \dontrun{
#' mylist <- list(foo = "bar", cleesh = NULL, baz = "qux", nitfol = NULL)
#' names(remove_nulls(mylist)) == c("foo", "baz")
#' }
remove_nulls <- function(l) {
  l[!sapply(l, is.null)]
}
