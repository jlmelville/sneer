#' Optimization Direction Methods
#'
#' The available direction methods that can be used by the optimization routines
#' in sneer.
#'
#' @examples
#' make_opt(direction = steepest_descent())
#'
#' @keywords internal
#' @name optimization_direction
#' @family sneer optimization direction methods
NULL

#' Steepest Descent Direction
#'
#' Factory function for creating an optimizer direction method.
#'
#' Creates a gradient descent direction method for use by the optimizer.
#' Uses the direction of steepest descent.
#'
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_direction_interface}}
#' for details on the functions and values defined for this method.
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
