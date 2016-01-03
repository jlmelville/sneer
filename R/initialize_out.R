# Functions for initializing the output embedding.

#' Output Initializers
#'
#' These methods deal with initializing the output coordinates of an embedding.
#' They have names that begin with 'out_from' to indicate that they are only
#' used to initialize the output data, not the input data.
#'
#' @seealso
#' The return value of the initializers should be assigned to the
#' \code{init_out} parameter of an embedding function such as
#' \code{\link{embed_dist}} and \code{\link{embed_prob}}.
#'
#' @examples
#'
#' \dontrun{
#' # pass to the init_out parameter of an embedding function
#'
#' # initialize from PCA
#' embed_dist(init_out = out_from_PCA(), ...)
#'
#' # initialize from small random differences
#' embed_prob(init_out = out_from_rnorm(sd = 1e-4), ...)
#' }
#' @keywords internal
#' @name output_initializers
#' @family sneer output initializers
NULL


#' Initialize Output Coordinates from PCA
#'
#' Output initialization function.
#'
#' The first \code{k} scores of the PCA of the input coordinates are used to
#' initialize the \code{k} dimensions of the embedded coordinates. Input
#' coordinates are centered but not scaled before the PCA is carried out.
#' This initialization can only be used if the input  data is in the form of
#' coordinates, not a distance matrix.
#'
#' @param k Number of output dimensions. For 2D visualization this is always 2.
#' @param verbose If \code{TRUE}, log information about the initialization.
#' @return Output initializer.
#' @export
#' @family sneer output intializers
#' @seealso \code{\link{embed_dist}} and \code{\link{embed_prob}}
#'   for how to use this function to configure an embedding.
#' @examples
#' \dontrun{
#' # Should be passed to the init_out argument of an embedding function:
#' embed_dist(init_out = out_from_PCA(), ...)
#' }
out_from_PCA <- function(k = 2, verbose = TRUE) {
  init_out(function(inp, out) {
    if (is.null(inp$xm)) {
      stop("PCA initialization is only possible with input coordinates")
    }
    out$ym <- scores_matrix(inp$xm, ncol = k, verbose = verbose)
    out
  })
}

#' Initialize Output Coordinates from Matrix
#'
#' Output initialization function.
#'
#' Creates output data and initialize coordinates from the specified matrix.
#'
#' @param k Number of output dimensions. For 2D visualization this is always 2.
#' @param init_config Configuration to initialize the coordinates from. Must
#' be a matrix with the same dimensions as the desired output coordinates.
#' @param verbose If \code{TRUE}, log information about the initialization.
#' @return Output initializer.
#' @export
#' @family sneer output intializers
#' @seealso \code{\link{embed_dist}} and \code{\link{embed_prob}}
#'   for how to use this function to configure an embedding.
#'
#' @examples
#' \dontrun{
#' # create a scores matrix using R PCA
#' pca_scores <- prcomp(iris[, 1:4], center = TRUE, retx = TRUE)$x[, 1:2]
#'
#' # Should be passed to the init_out argument of an embedding function:
#' embed_dist(init_out = out_from_matrix(pca_scores), ...)
#' }
out_from_matrix <- function(init_config, k = 2, verbose = TRUE) {
  init_out(function(inp, out) {
    n <- nrow(inp$dm)
    if (nrow(init_config) != n | ncol(init_config) != k) {
      stop("init_config does not match necessary configuration for ym")
    }
    if (verbose) {
      message("Initializing from matrix")
    }
    out$ym <- init_config
    out
  })
}

#' Initialize Output Coordinates from Normal Distribution
#'
#' Output initialization function.
#'
#' Creates output data and initializes embedding coordinates from a normal
#' distribution centered at zero.
#'
#' @param k Number of output dimensions. For 2D visualization this is always 2.
#' @param sd The standard deviation of the distribution.
#' @param verbose If \code{TRUE}, log information about the initialization.
#' @return Output initializer.
#' @export
#' @family sneer output intializers
#' @seealso \code{\link{embed_dist}} and \code{\link{embed_prob}}
#'   for how to use this function to configure an embedding.
#'
#' @examples
#' \dontrun{
#' # Should be passed to the init_out argument of an embedding function:
#'  embed_dist(init_out = out_from_rnorm(sd = 1e-4), ...)
#' }
out_from_rnorm <- function(k = 2, sd = 1e-4, verbose = TRUE) {
  init_out(function(inp, out){
    n <- nrow(inp$dm)
    message("Initializing from normal distribution with sd = ",
            formatC(sd))
    out$ym <- random_matrix_norm(n, ncol = k, sd = sd)
    out
  })
}

#' Initialize Output Coordinates from Uniform Distribution
#'
#' Output initialization function.
#'
#' Creates output data and initializes embedding coordinates from a random
#' uniform distribution. This is the initialization method suggested for use
#' by the authors of the \code{\link{nerv}} method.
#'
#' @param k Number of output dimensions. For 2D visualization this is always 2.
#' @param min Lower limit of the distribution.
#' @param max Upper limit of the distribution.
#' @param verbose If \code{TRUE}, log information about the initialization.
#' @return Output initializer.
#' @export
#' @references
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#' @family sneer output intializers
#' @seealso \code{\link{embed_dist}} and \code{\link{embed_prob}}
#'   for how to use this function to configure an embedding.
#'
#' @examples
#' \dontrun{
#' # Should be passed to the init_out argument of an embedding function:
#'  embed_dist(init_out = out_from_runif(min = -10, max = 10), ...)
#' }
out_from_runif <- function(k = 2, min = 0, max = 1, verbose = TRUE) {
  init_out(function(inp, out){
    n <- nrow(inp$dm)
    message("Initializing from uniform distribution between ", formatC(min),
            ", ", formatC(max))
    out$ym <- random_matrix_unif(n, ncol = k, min = min, max = max)
    out
  })
}

#' Output Initializer Wrapper
#'
#' Wrapper function to creates the input data list and runs the specific
#' initializer function provided as a parameter.
#'
#' @param initializer Initializer function.
#' @return Input data.
init_out <- function(initializer) {
  function(inp) {
    out <- list()
    out <- initializer(inp, out)
    flush.console()
    out
  }
}

#' PCA Scores Matrix
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

#' Random Matrix (Normal Distribution)
#'
#' Creates a matrix of normally distributed data with zero mean.
#'
#' @param nrow Number of rows of the matrix.
#' @param ncol Number of columns of the matrix.
#' @param sd Standard deviation of the distribution.
#' @return Random matrix with \code{nrow} rows and \code{ncol} columns.
#'
#' @examples
#' \dontrun{
#' # matrix with 5 rows, 3 columns and standard deviation of 0.1
#' xm <- random_matrix_norm(5, 3, 0.1)
#' # matrix with 100 rows
#' xm <- random_matrix_norm(100)
#' }
random_matrix_norm <- function(nrow, ncol = 2, sd = 1.0e-4) {
  matrix(rnorm(ncol * nrow, mean = 0, sd = sd), nrow = nrow)
}

#' Random Matrix (Uniform Distribution)
#'
#' Creates a matrix of uniformly distributed data.
#'
#' @param nrow Number of rows of the matrix.
#' @param ncol Number of columns of the matrix.
#' @param min Lower limit of the distribution.
#' @param max Upper limit of the distribution.
#' @return Random matrix with \code{nrow} rows and \code{ncol} columns.
#'
#' @examples
#' \dontrun{
#' # matrix with 5 rows, 3 columns distribution between -10 and 10
#' xm <- random_matrix_unif(nrow = 5, ncol = 3, min = -10, max = 10)
#' # 2D matrix with 100 rows and default range of 0 to 1
#' xm <- random_matrix_unif(100)
#' }
random_matrix_unif <- function(nrow, ncol = 2, min = 0, max = 1) {
  matrix(runif(n = ncol * nrow, min = min, max = max), nrow = nrow)
}
