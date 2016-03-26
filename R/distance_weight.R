# Distance weighting functions. Used to convert distances to similarities, which
# are in turn converted to probabilities in probability-based embedding. These
# functions work on the squared distances.

#' Exponential Weighted Similarity
#'
#' A similarity function for probability-based embedding.
#'
#' The weight matrix, \eqn{W} is generated from the squared distances matrix,
#' \eqn{D^2}{D2} by:
#'
#' \deqn{W = e^{-\beta D^2}}{exp(-beta * D2)}
#'
#' @param d2m Matrix of squared distances.
#' @param beta exponential parameter.
#' @return Weight matrix.
#' @family sneer weight functions
#' @export
exp_weight <- function(d2m, beta = 1) {
  exp(-beta * d2m)
}
attr(exp_weight, "type") <- "symm"

#' Exponential (Distance) Weighted Similarity
#'
#' A similarity function for probability-based embedding.
#'
#' Applies exponential weighting to the distances, rather than the square of
#' the distances. Included so results can be compared with the implementation
#' of t-SNE in the
#' \href{https://cran.r-project.org/web/packages/tsne/index.html}{R tsne}
#' package.
#'
#' The weight matrix, \eqn{W} is generated from the squared distances matrix,
#' \eqn{D^2}{D2} by:
#'
#' \deqn{W = e^{-\beta \sqrt{D^2}}}{W = exp(-beta * sqrt(D2))}
#'
#' @param d2m Matrix of squared distances.
#' @param beta Exponential precision.
#' @return Weight matrix.
#' @family sneer weight functions
#' @export
sqrt_exp_weight <- function(d2m, beta = 1) {
  exp(-beta * sqrt(d2m))
}
attr(sqrt_exp_weight, "type") <- "symm"

#' Student-t Distribution Similarity
#'
#' A similarity function for probability-based embedding.
#'
#' Applies weighting using the Student-t distribution with one degree of
#' freedom. Used in t-SNE.
#'
#' Compared to the exponential weighting this has a much heavier tail.
#' The weight matrix, \eqn{W} is generated from the squared distances matrix,
#' \eqn{D^2}{D2} by:
#' \deqn{W = \frac{1}{(1 + D^2)}}{W = 1/(1 + D2)}
#'
#' @param d2m Matrix of squared distances.
#' @return Weight matrix.
#' @family sneer weight functions
#' @export
tdist_weight <- function(d2m) {
  1 / (1 + d2m)
}
attr(tdist_weight, "type") <- "symm"


#' Heavy-Tailed Similarity
#'
#' A similarity function for probability-based embedding.
#'
#' Applies a "heavy-tailed" similarity function that represents
#' a generalization of the similarity functions used in SNE and t-SNE. The
#' heavy-tailedness is with respect to that of the exponential functon.
#' The weight matrix, \eqn{W} is generated from the squared distances matrix,
#' \eqn{D^2}{D2} by:
#' \deqn{W  = [(\alpha \beta D^2) + 1]^{-\frac{1}{\alpha}}}{W = ((alpha * beta * D2) + 1) ^ (-1 / alpha)}
#'
#' \eqn{\alpha \to 0}{alpha approaches 0}, the weighting function becomes
#' exponential, and behaves like \code{\link{exp_weight}}. At
#' \eqn{\alpha = 1}{alpha = 1}, the weighting function is the t-distribution
#' with one degree of freedom, \code{\link{tdist_weight}}. Intermediate
#' values provide an intermediate degree of tail heaviness.
#' The \eqn{\beta}{beta} parameter is equivalent to that in the
#' \code{\link{exp_weight}} function. This is set to one in methods suchs as
#' SSNE and t-SNE.
#'
#' @param d2m Matrix of squared distances.
#' @param beta The precision of the function. Becomes equivalent to the
#' precision in the Gaussian distribution of distances as \code{alpha}
#' approaches zero.
#' @param alpha Tail heaviness. Must be greater than zero.
#' @return Weight matrix.
#' @family sneer weight functions
#' @export
#' @references
#' Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
#' Heavy-tailed symmetric stochastic neighbor embedding.
#' In \emph{Advances in neural information processing systems} (pp. 2169-2177).
#' @examples
#' # make a matrix of squared distances
#' d2m <- dist(matrix(rnorm(12), nrow = 3)) ^ 2
#'
#' # exponential weighting
#' wm_exp <-   heavy_tail_weight(d2m, alpha = 1.5e-8)
#'
#' # t-distributed weighting
#' wm_tdist <- heavy_tail_weight(d2m, alpha = 0.0)
#'
#' # exponential weighting with a non-standard beta value
#' wm_expb2 <- heavy_tail_weight(d2m, alpha = 1.5e-8, beta = 2.0)
heavy_tail_weight <- function(d2m, beta = 1, alpha = 1.5e-8) {
  ((alpha * beta * d2m) + 1) ^ (-1 / alpha)
}
attr(heavy_tail_weight, "type") <- "symm"


#' Exponential Kernel Factory Function
#'
#' Similarity Kernel factory function.
#'
#' Creates a list implementing the exponential kernel function and gradient.
#'
#' @param beta Exponential parameter.
#' @return Exponential function and gradient.
#' @family sneer similiarity kernels
#' @export
exp_kernel <- function(beta = 1) {
  fn <- function(kernel, d2m) {
    exp_weight(d2m, beta = kernel$beta)
  }

  kernel <- list(
    fn = fn,
    gr = function(kernel, d2m) {
      exp_gr(d2m, beta = kernel$beta)
    },
    check_symmetry = function(kernel) {
      if (length(kernel$beta) > 1) {
        attr(kernel$fn, "type") <- "asymm"
      }
      else {
        attr(kernel$fn, "type") <- "symm"
      }
      kernel
    },
    beta = beta
  )
  check_symmetry(kernel)
}

#' Exponential Gradient
#'
#' Similarity Kernel Gradient.
#'
#' Calculates the gradient of the exponential function with respect to d2m,
#' the matrix of squared distances.
#'
#' @param d2m Matrix of squared distances.
#' @param beta exponential parameter.
#' @return Matrix containing the gradient of (with respect to \code{d2m}).
exp_gr <- function(d2m, beta = 1) {
  -beta * exp_weight(d2m, beta = beta)
}

#' t-Distribution Kernel Factory Function
#'
#' Similarity Kernel factory function.
#'
#' Creates a list implementing the t-distributed kernel function and gradient.
#'
#' @return t-Distributed function and gradient.
#' @family sneer similiarity kernels
#' @export
tdist_kernel <- function() {
  fn <- function(kernel, d2m) {
    tdist_weight(d2m)
  }
  attr(fn, "type") <- attr(tdist_weight, "type")

  list(
    fn = fn,
    gr = function(kernel, d2m) {
      tdist_gr(d2m)
    }
  )
}

#' Exponential Gradient
#'
#' t-Distributed Kernel Gradient.
#'
#' Calculates the gradient of the Student-t distribution with one degree of
#' freedom with respect to d2m, the matrix of squared distances.
#'
#' @param d2m Matrix of squared distances.
#' @return Matrix containing the gradient (with respect to \code{d2m}).
tdist_gr <- function(d2m) {
  -(tdist_weight(d2m) ^ 2)
}

#' Heavy Tailed Kernel Factory Function
#'
#' Similarity Kernel factory function.
#'
#' Creates a list implementing a heavy tailed (compared to an exponential)
#' function and gradient.
#'
#' @param beta The bandwidth of the function. Becomes equivalent to the
#' precision of the exponential distribution of squared distances as
#' \code{alpha} approaches zero.
#' @param alpha Tail heaviness. Must be greater than zero.
#' @return Heavy tailed function and gradient.
#' @family sneer similiarity kernels
#' @export
heavy_tail_kernel <- function(beta = 1, alpha = 0) {
  fn <- function(kernel, d2m) {
    heavy_tail_weight(d2m, beta = kernel$beta, alpha = kernel$alpha)
  }

  kernel <- list(
    fn = fn,
    gr = function(kernel, d2m) {
      heavy_tail_gr(d2m, beta = kernel$beta, alpha = kernel$alpha)
    },
    beta = beta,
    alpha = clamp(alpha, sqrt(.Machine$double.eps)),
    check_symmetry = function(kernel) {
      if (length(kernel$beta) > 1 || length(kernel$alpha) > 1) {
        attr(kernel$fn, "type") <- "asymm"
      }
      else {
        attr(kernel$fn, "type") <- "symm"
      }
      kernel
    }
  )
  kernel <- check_symmetry(kernel)
  kernel
}

#' Heavy Tail Kernel Gradient.
#'
#' Calculates the gradient of the Student-t distribution with one degree of
#' freedom with respect to d2m, the matrix of squared distances.
#'
#' @param d2m Matrix of squared distances.
#' @param beta The precision of the function. Becomes equivalent to the
#' precision in the Gaussian distribution of distances as \code{alpha}
#' approaches zero.
#' @param alpha Tail heaviness. Must be greater than zero.
#' @return Matrix containing the gradient (with respect to \code{d2m}).
heavy_tail_gr <- function(d2m, beta = 1, alpha = 1.5e-8) {
  -beta * heavy_tail_weight(d2m, beta = beta, alpha = alpha) ^ (alpha + 1)
}

#' Finite Difference Gradient of Kernel
#'
#' Calculates the gradient of a similarity kernel by finite difference.
#' Only intended for testing purposes.
#'
#' @param kernel A similarity kernel.
#' @param d2m Matrix of squared distances.
#' @param diff Step size to take in finite difference calculation.
#' @return Gradient matrix.
kernel_gr_fd <- function(kernel, d2m, diff = 1e-4) {
  d2m_fwd <- d2m + diff
  fwd <- kernel$fn(kernel, d2m_fwd)

  d2m_back <- d2m - diff
  back <- kernel$fn(kernel, d2m_back)

  (fwd - back) / (2 * diff)
}

#' Ensure the Kernel has the Correct Symmetry
#'
#' This function should be called when the parameters of a kernel (e.g.
#' the beta parameter of the exponential) are changed from a scalar to
#' a vector or (vice versa). Such a change may result in the symmetry of
#' the kernel changing from asymmetric to symmetric (and vice versa again).
#'
#' @param kernel A similarity kernel.
#' @return Kernel with the type attribute of the function correctly set for
#' its parameters.
check_symmetry <- function(kernel) {
  if (!is.null(kernel$check_symmetry)) {
    kernel <- kernel$check_symmetry(kernel)
  }
  kernel
}
