# output initialization
make_init_out <- function(k = 2, initial_config = NULL, stdev = 1e-04,
                          from_PCA = FALSE, from_input_name = NULL,
                          verbose = TRUE, mat_name = "ym") {
  init_out <- list()

  if (!is.null(initial_config) && is.matrix(initial_config)) {
    init_out$from_data <- function(inp, out) {
      n <- nrow(inp$xm)
      if (nrow(initial_config) != n | ncol(initial_config) != k) {
        stop("initial_config does not match necessary configuration for xm")
      }
      out[[mat_name]] <- initial_config
      out
    }
  } else if (from_PCA) {
    init_out$from_PCA <- function(inp, out) {
      out[[mat_name]] <- scores_matrix(inp$xm, ncol = k, verbose = verbose)
      out
    }
  } else if (!is.null(from_input_name)) {
    init_out$from_input <- function(inp, out) {
      if (verbose) {
        message("Initializing out$", mat_name, " from inp$", from_input_name)
      }
      out[[mat_name]] <- inp[[from_input_name]]
      out
    }
  } else {
    init_out$from_random <- function(inp, out) {
      n <- nrow(inp$xm)
      out[[mat_name]] <- random_matrix(n, ncol = k, sd = stdev)
      out
    }
  }

  function(inp) {
    out <- list()
    for (name in names(init_out)) {
      out <- init_out[[name]](inp, out)
    }
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
#' # first two components of PCA
#' scores <- scores_matrix(xm, ncol = 2)
#' # all scores
#' scores <- scores_matrix(xm)
scores_matrix <- function(xm, ncol = min(nrow(xm), base::ncol(xm)),
                          verbose = TRUE) {
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
#' @return A random matrix with \code{nrow} rows and \code{ncol} columns.
#'
#' @examples
#' # matrix with 5 rows, 3 columns and standard deviation of 0.1
#' xm <- random_matrix(5, 3, 0.1)
#' # matrix with 100 rows
#' xm <- random_matrix(100)
random_matrix <- function(nrow, ncol = 2, sd = 1.0e-4) {
  matrix(rnorm(ncol * nrow, mean = 0, sd = sd), nrow = nrow)
}

