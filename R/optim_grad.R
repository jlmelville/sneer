# Optimizer gradient methods.

#' Classical Gradient.
#'
#' Configuration function for optimizer gradient calculation.
#'
#' @return Classical gradient calculation method.
classical_gradient <- function() {
  classical_grad_pos
}

#' Nesterov Accelerated Gradient.
#'
#' Configuration function for optimizer gradient calculation.
#'
#' @return NAG calculation method.
#' @references
#' Sutskever, I., Martens, J., Dahl, G. and Hinton, G. E.
#' On the importance of momentum and initialization in deep learning.
#' 30th International Conference on Machine Learning, Atlanta, USA, 2013.
#' JMLR: W&CP volume 28.
nesterov_gradient <- function() {
  nesterov_grad_pos
}


#' Gradient calculation at current solution position.
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
#' \itemize{
#'  \item \code{km} Stiffness matrix.
#'  \item \code{gm} Gradient matrix.
#' }
classical_grad_pos <- function(opt, inp, out, method) {
  gradient(inp, out, method, opt$mat_name)
}

#' Nesterov Accelerated Gradient calculation.
#'
#' This function calculates the gradient at a position determined by applying
#' the momentum update to the current solution position.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return List containing:
#' \itemize{
#'  \item \code{km} Stiffness matrix.
#'  \item \code{gm} Gradient matrix.
#' }
#' @references
#' Sutskever, I., Martens, J., Dahl, G. and Hinton, G. E.
#' On the importance of momentum and initialization in deep learning.
#' 30th International Conference on Machine Learning, Atlanta, USA, 2013.
#' JMLR: W&CP volume 28.
nesterov_grad_pos <- function(opt, inp, out, method) {
  prev_update <- opt$update_method$update
  mu <- opt$update_method$momentum

  opt$update_method$update <- mu * prev_update
  new_out <- update_solution(opt, inp, out, method)

  gradient(inp, new_out, method, opt$mat_name)
}



