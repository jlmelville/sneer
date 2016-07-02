# Create Line Search Function
#
# @param opt Optimizer
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @param iter Iteration number.
# @param pm Direction vector.
# @param calc_gradient_default If \code{TRUE}, then the returned line
#  search function will calculate the cost function gradient and the line
#  function derivative by default.
# @return Line search function, which takes one parameter: the distance along
#  the line search and returns a list containing function evaluation information
#  at the distance. If \code{calc_gradient_default} is \code{TRUE} then gradient
#  information is included in the return value. Otherwise, this information can
#  be included by passing the optional \code{calc_gradient} argument to the line
#  search with the value \code{TRUE}.
make_phi_alpha <- function(opt, inp, out, method, iter, pm,
                           calc_gradient_default = FALSE) {
  out0 <- opt$gradient$calculate_position(opt, inp, out, method, iter)
  y0 <- out0[[opt$mat_name]]
  function(alpha, calc_gradient = calc_gradient_default) {
    y_alpha <- y0 + (alpha * pm)
    out_alpha <- set_solution(inp, y_alpha, method)
    f <- calculate_cost(method, inp, out_alpha)

    step <- list(
      alpha = alpha,
      f = f
    )

    if (calc_gradient) {
      step$df <- opt$gradient$calculate(opt, inp, out_alpha, method, iter)$gm
      step$d <- dot(step$df, pm)
      #      message(" p = ", format_vec(pm),
      #              " alpha = " , formatC(alpha),
      #              " f = ", formatC(step$f),
      #              " df = ", format_vec(step$df),
      #              " d = ", formatC(step$d))
    }
    step
  }
}
