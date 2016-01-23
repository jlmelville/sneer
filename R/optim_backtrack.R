#' Step Size By Backtracking Line Search
#'
#' Step size method.
#'
#' Uses the Backtracking line search method to find a step length that fulfills
#' the Wolfe conditions: the sufficient decrease condition (or Armijo conditon)
#' and the curvature condition.
#'
#' @param c1 Constant for the sufficient decrease condition.
#' @param rho Factor to decrease alpha by.
#' @param min_step_size If the step length falls below this value, exit the
#'  line search for this iteration and use this value.
#' @param max_step_size Initial step length to start the search at.
#' @param stop_at_min If \code{TRUE} and the step size was \code{min_step_size},
#'  terminate the optimization.
#' @return A step size method for use with an optimizer.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # step size method:
#' make_opt(step_size = backtracking(c1 = 0.1, rho = 0.8))
#' @references
#' Nocedal, J., & Wright, S. (2006).
#' \emph{Numerical optimization.}
#' Springer Science & Business Media.
#' @export
#' @family sneer optimization step size methods
backtracking <- function(c1 = 0.1, rho = 0.8,
                         max_step_size = 1,
                         min_step_size = .Machine$double.eps,
                         stop_at_min = TRUE) {
  list(
    init = function(opt, inp, out, method) {
      opt$step_size$value <- 1
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      opt$step_size$value <-
        backtracking_line_search(opt, inp, out, method, iter,
                                 alpha = opt$step_size$max_step_size,
                                 c1 = 1.e-1, rho = 0.8,
                                 min_step_size = min_step_size)
      list(opt = opt)
    },
    validate = function(opt, inp, out, proposed_out, method, iter) {
      ok <- TRUE
      if (opt$step_size$value <= min_step_size && stop_at_min) {
        opt$stop_early <- TRUE
        ok <- FALSE
      }
      opt$step_size$max_step_size <- opt$step_size$max_step_size * (1 + rho)
      list(ok = ok, opt = opt)
    },
    max_step_size = max_step_size
  )
}

#' Backtracking Line Search
#'
#' Finds a step length that satisifes the Wolfe conditions.
#'
#' This function implements backtracking line search to find a step length
#' which fulfils both the curvature and sufficient decrease (aka Armijo)
#' conditions. It starts the line search at a step length given by the
#' \code{alpha} parameter, and if the sufficient decrease condition is not
#' met, decreases the step length by a factor of \code{rho} until it does.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @param iter Iteration number.
#' @param alpha Initial step size.
#' @param c1 Constant for the sufficient decrease condition.
#' @param rho Factor to decrease alpha by.
#' @param min_step_size Mininum step size.
#' @return Step length satisfying the Wolfe conditions.
#' @references
#' Nocedal, J., & Wright, S. (2006).
#' \emph{Numerical optimization.}
#' Springer Science & Business Media.
backtracking_line_search <- function(opt, inp, out, method, iter,
                                     alpha = 1, c1 = 1.e-1, rho = 0.8,
                                     min_step_size = .Machine$double.eps) {
  pm <- opt$direction$value
  cgp <- c1 * sum(opt$gm * pm)
  out0 <- opt$gradient$calculate_position(opt, inp, out, method, iter)
  y0 <- out0[[opt$mat_name]]

  phi_0 <- cost_step_length(opt, inp, method, y0, pm, 0)
  phi_alpha <- cost_step_length(opt, inp, method, y0, pm, alpha)
  if (is.na(phi_alpha)) {
    stop("phi_alpha is NA")
  }
  if (is.na(alpha)) {
    stop("alpha is NA")
  }
  if (any(is.na(opt$gm))) {
    stop("NA in gm")
  }
  if (any(is.na(pm))) {
    stop("NA in pm")
  }
  if (is.na(cgp)) {
    stop("cgp is NA")
  }

  while (phi_alpha > phi_0 + (alpha * cgp) && alpha > min_step_size) {
    alpha <- rho * alpha
    phi_alpha <- cost_step_length(opt, inp, method, y0, pm, alpha)
  }
  #message(iter, ": alpha = ", formatC(alpha), " |gm| = ", formatC(length_vec(opt$gm)))

  alpha
}

#' Cost Evaluated at a Step Length Along a Search Direction
#'
#' This function calculates the cost function when the solution is updated
#' based on moving a specified step length along a search direction.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param method Embedding method.
#' @param y0 Coordinate start point of the line search.
#' @param pm Direction vector to search along.
#' @param alpha Step length along the direction.
#' @return Cost at the location given by \code{y0 + alpha * pm}.
cost_step_length <- function(opt, inp, method, y0, pm, alpha) {
  y_alpha <- y0 + (alpha * pm)
  out_alpha <- set_solution(opt, inp, y_alpha, method)
  if (any(is.na(out_alpha$ym))) {
    stop("NA in ym")
  }
  if (any(is.na(out_alpha$wm))) {
    stop("NA in wm")
  }
  if (any(is.na(out_alpha$qm))) {
    stop("NA in qm")
  }
  if (any(is.na(out_alpha$zm))) {
    stop("NA in zm")
  }
  if (is.na(out_alpha$kl_qz)) {
    stop("kl_qz is NA")
  }
  method$cost_fn(inp, out_alpha, method)
}
