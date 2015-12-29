#' Optimization Gradient Methods
#'
#' The available gradient methods that can be used by the optimization routines
#' in sneer.
#'
#' @examples
#' make_opt(gradient = classical_gradient())
#' make_opt(gradient = nesterov_gradient())
#' @keywords internal
#' @name optimization_gradient
#' @family sneer optimization gradient methods
NULL

#' Classical Gradient
#'
#' Factory function for creating an optimizer gradient method.
#'
#' Calculates the gradient at the current location of the solution.
#'
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_gradient_interface}}
#' for details on the functions and values defined for this method.
#'
#' @return Classical gradient calculation method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # gradient method:
#' make_opt(gradient = classical_gradient())
#' @family sneer optimization gradient methods
#' @export
classical_gradient <- function() {
  list(
    calculate = classical_grad_pos
  )
}


#' Nesterov Accelerated Gradient
#'
#' Factory function for creating an optimizer gradient method.
#'
#' Calculates the gradient according to the Nesterov Accelerated Gradient
#' method.
#'
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_gradient_interface}}
#' for details on the functions and values defined for this method.
#'
#' @return NAG method for gradient calculation.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # gradient method:
#' make_opt(gradient = nesterov_gradient())
#' @references
#' Sutskever, I., Martens, J., Dahl, G. and Hinton, G. E.
#' On the importance of momentum and initialization in deep learning.
#' 30th International Conference on Machine Learning, Atlanta, USA, 2013.
#' JMLR: W&CP volume 28.
#' @family sneer optimization gradient methods
#' @export
nesterov_gradient <- function() {
  list(
    calculate = nesterov_grad_pos
  )
}


#' Classical Gradient Calculation
#'
#' If the solution is currently at \code{out$ym}, this function calculates the
#' gradient is calculated at this position. Contrast this with Nesterov
#' Accelerated Gradient Descent.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return List containing:
#'  \item{\code{km}}{Stiffness matrix.}
#'  \item{\code{gm}}{Gradient matrix.}
classical_grad_pos <- function(opt, inp, out, method) {
  gradient(inp, out, method, opt$mat_name)
}

#' Nesterov Accelerated Gradient Calculation
#'
#' This function calculates the gradient at a position determined by applying
#' the momentum update to the current solution position.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return List containing:
#'  \item{\code{km}}{Stiffness matrix.}
#'  \item{\code{gm}}{Gradient matrix.}
#' @references
#' Sutskever, I., Martens, J., Dahl, G. and Hinton, G. E.
#' On the importance of momentum and initialization in deep learning.
#' 30th International Conference on Machine Learning, Atlanta, USA, 2013.
#' JMLR: W&CP volume 28.
nesterov_grad_pos <- function(opt, inp, out, method) {
  prev_update <- opt$update$value
  mu <- opt$update$momentum

  opt$update$value <- mu * prev_update
  new_out <- update_solution(opt, inp, out, method)

  gradient(inp, new_out, method, opt$mat_name)
}
