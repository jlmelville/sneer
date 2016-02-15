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
      opt$step_size$max <- max_step_size
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      pm <- opt$direction$value
      phi_alpha <- make_phi_alpha(opt, inp, out, method, iter, pm)

      opt$step_size$value <-
        backtracking_line_search(phi_alpha,
                                 alpha = opt$step_size$max,
                                 c1 = c1, rho = rho,
                                 min_step_size = min_step_size)
      list(opt = opt)
    },
    validate = function(opt, inp, out, proposed_out, method, iter) {
      ok <- TRUE
      if (opt$step_size$value <= min_step_size && stop_at_min) {
        opt$stop_early <- TRUE
        ok <- FALSE
      }
      list(ok = ok, opt = opt)
    },
    after_step = function(opt, inp, out, new_out, method, ok, iter) {
      opt$step_size$max <- min(1, opt$step_size$value * 1.01)
      list(opt = opt)
    },
    max = max_step_size
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
#' @param phi_alpha Line function.
#' @param alpha Initial step size.
#' @param c1 Constant for the sufficient decrease condition.
#' @param rho Factor to decrease alpha by.
#' @param min_step_size Mininum step size.
#' @return Step length satisfying the Wolfe conditions.
#' @references
#' Nocedal, J., & Wright, S. (2006).
#' \emph{Numerical optimization.}
#' Springer Science & Business Media.
backtracking_line_search <- function(phi_alpha,
                                     alpha = 1,
                                     c1 = 1.e-1, rho = 0.8,
                                     min_step_size = .Machine$double.eps) {

  s0 <- phi_alpha(0, calc_gradient = TRUE)
  sa <- phi_alpha(alpha)

  while (!armijo_oks(s0, sa, c1) && alpha > min_step_size) {
    alpha <- rho * alpha
    sa <- phi_alpha(alpha)
#    message("alpha = ", formatC(alpha),
#            " f0 = ", formatC(s0$f),
#            " f = ", formatC(sa$f))
  }

  alpha
}



