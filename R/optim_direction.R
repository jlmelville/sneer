#' Optimizer direction methods.
#'
#' Part of the optimizer that finds the direction of descent.
#'
#' @section Interface:
#' A direction method is a list containing:
#' \describe{
#' \item{\code{init(opt, inp, out, method)}}{Optional initialization function
#' with the following arguments:
#'  \describe{
#'    \item{\code{opt}}{Optimizer.}
#'    \item{\code{inp}}{Input data.}
#'    \item{\code{out}}{Output data.}
#'    \item{\code{method}}{Embedding method.}
#'  }
#'  The function should set any needed state on \code{opt$direction} and return
#'  a list containing:
#'  \describe{
#'    \item{\code{opt}}{Optimizer containing initialized \code{direction}.}
#'  }
#' }
#' \item{\code{calculate(opt, inp, out, method, iter)}}{Calculation function
#' with the following arguments:
#'  \describe{
#'    \item{\code{opt}}{Optimizer.}
#'    \item{\code{inp}}{Input data.}
#'    \item{\code{out}}{Output data.}
#'    \item{\code{method}}{Embedding method.}
#'    \item{\code{iter}}{Iteration number.}
#'  }
#'  The function should \code{opt$direction$value} with the current direction
#'  of descent and return a list containing:
#'  \describe{
#'    \item{\code{opt}}{Optimizer containing updated \code{direction$value}.}
#'  }
#' }
#' \item{\code{value}}{The current direction. It should be a matrix with
#' the same dimensions as the gradient.}
#' }
#' @keywords internal
#' @name optimizer_direction
NULL


#' Steepest Descent direction.
#'
#' Optimizer direction method.
#'
#' Creates a gradient descent direction method for use by the optimizer.
#' Uses the direction of steepest descent.
#'
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimizer_direction}} for details
#' on the functions and values defined for this method.
#'
#' @return Steepest Descent direction method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # direction method:
#' make_opt(direction = steepest_descent())
#' @export
#' @family sneer optimization direction methods
steepest_descent <- function() {
  list(
    calculate = function(opt, inp, out, method, iter) {
      opt$direction$value <- -opt$gm
      list(opt = opt)
    }
  )
}
