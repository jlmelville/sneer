# Perplexity Calculations. Used in creating the input probabilities in
# probability-based embedding.

#' Summarize Parameter Search
#'
#' Provides summary of the distribution of the beta parameter used to
#' generate input probabilities in SNE.
#'
#' Mainly useful for debugging. Also expresses beta as sigma, i.e. the
#' Gaussian bandwidth \eqn{\frac{1}{\sqrt{2\beta}}}{1/sqrt(2*beta)}.
#'
#' @param betas Vector of parameters
summarize_betas <- function(betas) {
  summarize(prec_to_bandwidth(betas), "sigma")
  summarize(betas, "beta")
}

#' Convert Precisions to Sigma (Bandwidths)
#'
#' Gaussian-type functions have a parameter associated with them
#' that is either thought of as a "precision", i.e. how "tight" the distribution
#' is, or its counterpart, the bandwidth, which measure how spread-out the
#' distribution is. For gaussian functions, a direct comparison can be made
#' with the standard deviation of the function. Exact definitions of the
#' parameters differ in different discussions. In sneer, the gaussian function
#' is defined as:
#'
#' \deqn{e(-\beta D^2)}{exp(-beta * D^2)}
#'
#' where \eqn{\beta}{beta} is the precision, or alternatively:
#'
#' \deqn{e^{-\frac{D^2}{2\sigma^2}}}{exp[-(D ^ 2)/(2 * sigma ^ 2)]}
#'
#' so that \eqn{\sigma = \frac{1}{\sqrt{2\beta}}}{sigma = 1/sqrt(2*beta)}
#'
#' This function performs the conversion of the parameter between precision and
#' bandwidth.
#'
#' @param prec Precision (beta).
#' @return Bandwidth (sigma).
prec_to_bandwidth <- function(prec) {
  1 / sqrt(2 * prec)
}

#' Find Row Probabilities by Perplexity Bisection Search
#'
#' For each row, finds the value of beta which generates the probability
#' with the desired perplexity.
#'
#' @param dm Distance matrix.
#' @param perplexity Target perplexity value.
#' @param weight_fn Function which maps squared distances to weights. Should
#' have the following signature: \code{function(d2m, beta)}
#' @param tol Convergence tolerance for perplexity.
#' @param max_iters Maximum number of iterations to carry out the search.
#' @param verbose If \code{TRUE}, logs information about the beta values.
#' @return List with the following members:
#'  \item{\code{pm}}{Row probability matrix. Each row is a probability
#'  distribution with a perplexity within \code{tol} of \code{perplexity}.}
#'  \item{\code{beta}}{Matrix of beta parameters used with \code{weight_fn} that
#'  generated \code{pm}.}
d_to_p_perp_bisect <- function(dm, perplexity = 15, weight_fn, tol = 1e-05,
                               max_iters = 50, verbose = TRUE) {
  d2m <- dm ^ 2
  n <- nrow(d2m)

  pm <- matrix(0, n, n)
  beta <- rep(1, n)
  for (i in 1:n) {
    d2mi <- d2m[i, , drop = FALSE]
    result <- find_beta(d2mi, i, perplexity, beta[i], weight_fn, tol, max_iters)
    pm[i,] <- result$pr
    beta[i] <- result$beta
  }

  attr(pm, 'type') <- "row"
  if (verbose) {
    summarize_betas(beta)
    summarize(pm, "P")
  }
  list(pm = pm, beta = beta)
}

#' Find Beta Parameter for One Row of Matrix
#'
#' Find the value of beta that produces a specific perplexity when a row of
#' a distance matrix is converted to probabilities.
#'
#' @param d2mi Row of a squared distance matrix.
#' @param i Index of the row in the squared distance matrix.
#' @param perplexity Target perplexity value.
#' @param beta_init Initial guess for beta.
#' @param weight_fn Function which maps squared distances to weights. Should
#' have the following signature: \code{function(D2i, beta)}
#' @param tol Convergence tolerance for perplexity.
#' @param max_iters Maximum number of iterations to carry out the search.
#' @return List with the following members:
#'  \item{\code{pm}}{Matrix with one row containing a probability distribution
#'  with a perplexity within \code{tol} of \code{perplexity}.}
#'  \item{\code{beta}}{Beta parameter used with \code{weight_fn} that generated
#'  \code{pm}.}
#'  \item{\code{perp}}{Final perplexity of the probability, differing from
#'  \code{perplexity} only if \code{max_iters} was exceeded.}
find_beta <- function(d2mi, i, perplexity, beta_init = 1,
                      weight_fn, tol = 1e-05, max_iters = 50) {

  h_base <- exp(1)
  fn <- make_objective_fn(d2mi, i, weight_fn = weight_fn,
                          perplexity = perplexity, h_base = h_base)

  result <- root_bisect(fn = fn, tol = tol, max_iters = max_iters,
                        x_lower = 0, x_upper = Inf, x_init = beta_init)

  list(pr = result$best$pr, perplexity = h_base ^ result$best$h,
       beta = result$x)
}

#' Find Root of Function by Bisection
#'
#  Bisection method to find root of f(x) = 0 given two values which bracket
#' the root.
#'
#' @param fn Function to optimize. Should take one scalar numeric value and
#' return a list containing at least one element called \code{y}, which should
#' contain the scalar numeric value. The list may contain other elements, so
#' that other useful values can be returned from the search routine.
#' @param tol Tolerance. If \eqn{abs(y_target - y) < tol} then the search will
#' terminate.
#' @param max_iters Maximum number of iterations to search for.
#' @param x_lower Lower bracket of x.
#' @param x_upper Upper bracket of x.
#' @param x_init Initial value of x.
#' @return a list containing:
#'  \item{\code{x}}{Optimized value of x.}
#'  \item{\code{y}}{Value of y at convergence or \code{max_iters}}
#'  \item{\code{iter}}{Number of iterations at which convergence was reached.}
#'  \item{\code{best}}{Return value of calling \code{fn} with the optimized
#'  value of \code{x}.}
root_bisect <- function(fn, tol = 1.e-5, max_iters = 50, x_lower,
                        x_upper, x_init = (x_lower + x_upper) / 2) {
  result <- fn(x_init)
  value <- result$value
  bounds <- list(lower = min(x_lower, x_upper), upper = max(x_lower, x_upper),
                 mid = x_init)
  iter <- 0
  while (abs(value) > tol && iter < max_iters) {
    bounds <- improve_guess(bounds, sign(value))
    result <- fn(bounds$mid)
    value <- result$value
    iter <- iter + 1
  }
  list(x = bounds$mid, y = value, best = result, iter = iter)
}

#' Narrows Bisection Bracket Range
#'
#' @param bracket List representing the bounds of the parameter search.
#' Contains three elements: \code{lower}: lower value, \code{upper}: upper
#' value, \code{mid}: the midpoint.
#' @param sgn Sign of the value of the function evaluted at \code{bracket$mid}.
#' @return Updated \code{bracket} list.
improve_guess <- function(bracket, sgn) {
  mid <- bracket$mid
  if (sgn > 0) {
    bracket$lower <- mid
    if (is.infinite(bracket$upper)) {
      mid <- mid * 2
    } else {
      mid <- (mid + bracket$upper) / 2
    }
  } else {
    bracket$upper <- mid
    if (is.infinite(bracket$lower)) {
      mid <- mid / 2
    } else {
      mid <- (mid + bracket$lower) / 2
    }
  }
  bracket$mid <- mid
  bracket
}

#' Objective Function for Parameter Search
#'
#' Create callback that can be used as an objective function for perplexity
#' parameter search.
#'
#' @param d2r Row of a squared distance matrix.
#' @param i Index of the row in the matrix.
#' @param weight_fn Function that converts squared distances to weights. Should
#' have the signature \code{function(D2, beta)}
#' @param perplexity the target perplexity the probability generated by the
#' weight function should produce.
#' @param h_base the base of the logarithm to use for Shannon Entropy
#' calculations.
#' @return Callback function for use in parameter search routine. The callback
#' has the signature \code{fn(beta)} where \code{beta} is the
#' parameter being optimized, and returns a list
#' containing:
#'  \item{\code{value}}{Difference between the Shannon Entropy for the value
#'   of beta passed as an argument and the target Shannon Entropy.}
#'  \item{\code{h}}{Shannon Entropy for the value of beta passed as an
#' argument.}
#'  \item{\code{pm}}{Probabilities generated by the weighting function.}
make_objective_fn <- function(d2r, i, weight_fn, perplexity, h_base = exp(1)) {
  h_target <- log(perplexity, h_base)

  weight_fn_param <- function(beta) {
    wr <- weight_fn(d2r, beta)
    wr[1, i] <- 0
    wr
  }
  function(beta) {
    wr <- weight_fn_param(beta)
    pr <- weights_to_prow(wr)
    pr <- clamp(pr)
    h <- shannon_entropy_rows(pr, h_base)
    list(value = h - h_target, pr = pr, h = h)
  }
}

#' Shannon Entropy per Row of a Matrix
#'
#' Each row of the matrix should consist of probabilities summing to 1.
#'
#' @param pm Matrix of probabilities.
#' @param base Base of the logarithm to use in the entropy calculation.
#' @return Vector of Shannon entropies, one per row of the matrix.
shannon_entropy_rows <- function(pm, base = 2) {
  -apply(pm * log(pm, base), 1, sum)
}


