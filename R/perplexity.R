# perplexity.R
# Perplexity Calculations

#' Log summary info on the distribution of the beta parameter used to
#' generate input probabilities in SNE.
#'
#' Mainly useful for debugging. Also expresses beta as sigma, i.e. the
#' Gaussian bandwidth 1/sqrt(2*beta).
#'
#' @param betas Vector of parameters
summarize_betas <- function(betas) {
  summarise(1 / sqrt(2 * betas), "sigma")
  summarise(betas, "beta")
}

#' Convert distance matrix to row probability matrix.
#'
#' For each row, finds the value of beta which generates the probability
#' with the desired perplexity.
#'
#' @param D Distance matrix.
#' @param perplexity Target perplexity value.
#' @param weight_fn Function which maps squared distances to weights. Should
#' have the following signature: \code{function(D2i, beta)}
#' @param tol Convergence tolerance for perplexity.
#' @param max_iters Maximum number of iterations to carry out the search.
#' @param verbose If TRUE, logs information about the beta values.
#' @return List with the following members: \code{pm} a row probability matrix
#' where each row is a probability distribution with a perplexity within
#' \code{tol} of \code{perplexity}, \code{beta} the beta parameter used with
#' \code{weight_fn} that generated \code{pm}.
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

  if (verbose) {
    summarize_betas(beta)
  }
  list(pm = pm, beta = beta)
}

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
#' @return List with the following members: \code{pm} a 1-row matrix containing
#' a probability distribution with a perplexity within \code{tol} of
#' \code{perplexity}, \code{beta} the beta parameter used with \code{weight_fn}
#' that generated \code{pm}, \code{perp} the final perplexity of the probability,
#' differing from \code{perplexity} only if \code{max_iters} was exceeded.
find_beta <- function(d2mi, i, perplexity, beta_init = 1,
                      weight_fn, tol = 1e-05, max_iters = 50) {

  h_base <- exp(1)
  fn <- make_objective_fn(d2mi, i, weight_fn = weight_fn,
                          perplexity = perplexity, h_base = h_base)

  result <- root_bisect(fn = fn, tol = tol, max_iters = max_iters,
                        x_neg = Inf, x_pos = 0, x_init = beta_init)

  list(pr = result$best$pr, perplexity = h_base ^ result$best$h,
       beta = result$x)
}

#' Bisection method to find root of f(x) = 0 given two values which bracket
#' the root.
#'
#' @param fn Function to optimize. Should take one scalar numeric value and
#' return a list containing at least one element called \code{y}, which should
#' contain the scalar numeric value. The list may contain other elements, so
#' that other useful values can be returned from the search routine.
#' @param tol Tolerance. If \code{abs(y_target - y) < tol} then the search will
#' terminate.
#' @param max_iters Maximum number of iterations to search for.
#' @param lower Lower bracket of x.
#' @param upper Upper bracket of x.
#' @return a list containing: \code{x} optimized value of x, \code{y} value of y
#' at convergence or \code{max_iters}, \code{iter} number of iterations at which
#' convergence was reached, \code{best} list containing all values returned by
#' calling \code{fn} with \code{x}.
root_bisect <- function(fn, tol = 1.e-5, max_iters = 50, x_pos,
                        x_neg, x_init = (x_pos + x_neg) / 2) {
  result <- fn(x_init)
  value <- result$value
  bounds <- list(lower = min(x_neg, x_pos), upper = max(x_neg,x_pos),
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

#' update_methods the bisection bracket for the parameter based on the discrepancy
#' between the output and target value.
#'
#' @param bracket List representing the bounds of the parameter search.
#' Contains three elements: \code{lower}: lower value, \code{upper}: upper value,
#' \code{mid}: the midpoint.
#' @param sgn Sign of the value of the function evaluted at \code{bracket$mid}.
#' @return the updated \code{bracket} list.
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

#' Create a one-parameter function that can be used as an objective function
#' for creating a probability distribution with a target perplexity.
#'
#' @param d2r Row of a squared distance matrix.
#' @param i Index of the row in the matrix.
#' @param weight_fn Function that converts squared distances to weights. Should
#' have the signature \code{function(D2, beta)}
#' @param perplexity the target perplexity the probability generated by the
#' weight function should produce.
#' @param h_base the base of the logarithm to use for Shannon Entropy
#' calculations.
#' @return a one-parameter function with the signature \code{function(beta)}.
#' It returns a list containing: \code{value} the difference between the Shannon
#' Entropy for the value of beta passed as an argument and the target Shannon
#' Entropy, \code{h} the Shannon Entropy for the value of beta passed as an
#' argument, \code{pm} the probabilities generated by the weighting function.
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
    h <- shannon_entropy_row(pr, h_base)
    list(value = h - h_target, pr = pr, h = h)
  }
}

#' Calculates the Shannon Entropy per row of a matrix.
#'
#' Each row of the matrix should consist of probabilities summing to 1.
#'
#' @param pm the matrix of probabilities.
#' @param base the base of the logarithm to use in the entropy calculation.
#' @return a vector of Shannon entropies, one per row of the matrix.
#'
shannon_entropy_row <- function(pm, base = 2) {
  -apply(pm * log(pm, base), 1, sum)
}
