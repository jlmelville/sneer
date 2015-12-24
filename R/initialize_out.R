# Functions for initializing the output embedding.

#' Create an output initialization callback.
#'
#' Factory function that creates a callback that used by the embedding routine.
#' When invoked, it will initialize the output data.
#'
#' @param k Number of output dimensions. For 2D visualization this is always 2.
#' @param init_config Input data to initialize the coordinates from. Must
#' be a matrix with the same dimensions as the desired output coordinates.
#' @param stdev The standard deviation of the gaussian used to generate the
#' output coordinates. Used only if \code{from_PCA} is \code{FALSE},
#' \code{init_config} is \code{NULL} and \code{from_input_name} is
#' \code{NULL}.
#' @param from_PCA If \code{TRUE}, then the first \code{k} scores of the PCA
#' of the input coordinates are used to initialize the embedded coordinates.
#' Input coordinates are centered but not scaled before the PCA is carried out.
#' This option can only be used if the input data is in the form of coordinates,
#' and not a distance matrix.
#' @param from_input_name Name of a matrix in the input data list \code{inp}
#' which contains suitable initialization data. Must be a matrix with the same
#' dimensions as the desired output coordinates.
#' @param mat_name Name of the matrix on the output list to contain the
#' initialized coordinates. Ensure that if you change this from the default that
#' other callbacks that need this information (e.g. optimization routines,
#' plotting function in reporter callbacks) are also passed the same value.
#' @param verbose If \code{TRUE} information about the initialization will be
#' logged to screen.
#' @return Callback to be used by the embedding routine to initialize
#' the output data. The callback has the signature \code{init_out(inp)} where
#' \code{inp} is the initialized input data list. The return value of the
#' callback is the initialized output data, a list containing:
#' \item{\code{ym}}{A matrix containing the initialized output coordinates. The
#'  name of the matrix can be changed by setting the \code{mat_name} parameter.}
#' Depending on the type of embedding carried out, the \code{out} list will
#' accumulate other auxiliary data associated with the embedded coordinates,
#' e.g. distances, probabilities, divergences or other data required for
#' stiffness calculations.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding.
#' @examples
#' # Initialize from PCA scores matrix (normally a decent starting point and
#' # reproducible)
#' make_init_out(from_PCA = TRUE)
#'
#' # Initialize from random small distances (used in t-SNE paper)
#' make_init_out(stdev = 1e-4)
#'
#' # Should be passed to the init_out argument of an embedding function:
#' \dontrun{
#'  embed_sim(init_out = make_init_out(from_PCA = TRUE), ...)
#' }
#' @export
make_init_out <- function(k = 2, init_config = NULL, stdev = 1e-04,
                          from_PCA = FALSE, from_input_name = NULL,
                          mat_name = "ym", verbose = TRUE) {
  init_out <- list()

  if (!is.null(init_config) && is.matrix(init_config)) {
    init_out$from_data <- function(inp, out) {
      n <- nrow(inp$dm)
      if (nrow(init_config) != n | ncol(init_config) != k) {
        stop("init_config does not match necessary configuration for ym")
      }
      out[[mat_name]] <- init_config
      out
    }
  } else if (from_PCA) {
    init_out$from_PCA <- function(inp, out) {
      if (is.null(inp$xm)) {
        stop("PCA initialization is only possible with input coordinates")
      }
      out[[mat_name]] <- scores_matrix(inp$xm, ncol = k, verbose = verbose)
      out
    }
  } else if (!is.null(from_input_name)) {
    init_out$from_input <- function(inp, out) {
      n <- nrow(inp$dm)
      init_config <- inp[[from_input_name]]
      if (nrow(init_config) != n | ncol(init_config) != k) {
        stop("inp$", from_input_name, " does not match necessary configuration",
             " for ym")
      }
      if (verbose) {
        message("Initializing out$", mat_name, " from inp$", from_input_name)
      }
      out[[mat_name]] <- inp[[from_input_name]]
      out
    }
  } else {
    init_out$from_random <- function(inp, out) {
      n <- nrow(inp$dm)
      out[[mat_name]] <- random_matrix(n, ncol = k, sd = stdev)
      out
    }
  }

  function(inp) {
    out <- list()
    for (name in names(init_out)) {
      out <- init_out[[name]](inp, out)
    }
    flush.console()
    out
  }
}

#' Creates a matrix consisting of the PCA scores of the input.
#'
#' @param xm Matrix to carry out PCA on. No scaling is carried out.
#' @param ncol The number of score columns to include in the output matrix.
#' Cannot be larger than the smaller of the number of rows or columns. Columns
#' are included in order of decreasing eigenvalue.
#' @param verbose If true, then information about variance explained by the
#' chosen number of columns will be logged.
#' @return A column matrix of the PCA scores. The number of rows of the matrix
#' is the same as that of \code{xm}.
#'
#' @examples
#' \dontrun{
#' # first two components of PCA
#' scores <- scores_matrix(iris[, 1:4], ncol = 2)
#' # all scores
#' scores <- scores_matrix(iris[, 1:4])
#' }
scores_matrix <- function(xm, ncol = min(nrow(xm), base::ncol(xm)),
                          verbose = TRUE) {

  xm <- scale(xm, center = TRUE, scale = FALSE)
  # do SVD on xm directly rather than forming covariance matrix
  ncomp <- ncol
  sm <- svd(xm, nu = ncomp, nv = 0)
  dm <- diag(c(sm$d[1:ncomp]))
  if (verbose) {
    # calculate eigenvalues of covariance matrix from singular values
    lambda <- (sm$d ^ 2) / (nrow(xm) - 1)
    varex <- sum(lambda[1:ncomp]) / sum(lambda)
    message("PCA: ", ncomp, " components explained ",
            formatC(varex * 100), "% variance")
  }
  sm$u %*% dm
}

#' Creates a matrix of normally distributed data with zero mean.
#'
#' @param nrow The number of rows of the matrix.
#' @param ncol The number of columns of the matrix.
#' @param sd The standard deviation of the distribution.
#' @return Random matrix with \code{nrow} rows and \code{ncol} columns.
#'
#' @examples
#' \dontrun{
#' # matrix with 5 rows, 3 columns and standard deviation of 0.1
#' xm <- random_matrix(5, 3, 0.1)
#' # matrix with 100 rows
#' xm <- random_matrix(100)
#' }
random_matrix <- function(nrow, ncol = 2, sd = 1.0e-4) {
  matrix(rnorm(ncol * nrow, mean = 0, sd = sd), nrow = nrow)
}
