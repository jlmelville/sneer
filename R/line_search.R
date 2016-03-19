make_phi_alpha <- function(opt, inp, out, method, iter, pm) {
  out0 <- opt$gradient$calculate_position(opt, inp, out, method, iter)
  y0 <- out0[[opt$mat_name]]
  function(alpha, calc_gradient = FALSE) {
    y_alpha <- y0 + (alpha * pm)
    out_alpha <- set_solution(opt, inp, y_alpha, method)
    f <- method$cost$fn(inp, out_alpha, method)

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
