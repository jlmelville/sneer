# Optimizer direction methods.

#' Steepest Descent direction.
#'
#' Creates a gradient descent direction method for use by the optimizer.
#' Uses the direction of steepest descent.
#'
#' @return Steepest Descent direction method. A list consisting of:
#' \itemize{
#'  \item \code{get_direction} Function invoked by the optimizer to find the direction
#'  to move in the gradient descent part of the solution update.
#' }
#' @export
steepest_descent <- function() {
  list(
    get_direction = function(opt, inp, out, method, iter) {
      list(direction = -opt$gm)
    }
  )
}
