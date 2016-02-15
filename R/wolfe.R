#' Armijo Rule (or Sufficient Decrease Condition)
#'
#' @param phi_0 Value of phi at alpha = 0
#' @param dphi_0 the directional derivative at alpha = 0
#' @param alpha the step length.
#' @param phi_a the value of phi at \code{alpha}.
#' @param c1 the sufficient decrease constant.
#' @value \code{TRUE} if the step \code{alpha} represents a sufficient decrease.
armijo_ok <- function(phi_0, dphi_0, alpha, phi_a, c1) {
#  message("phia = ", formatC(phi_a),
#          " phi0 = ", formatC(phi_0),
#          " cgp = ", formatC(c1 * alpha * dphi_0),
#          " phi_max = ", formatC(phi_0 + c1 * alpha * dphi_0))

  phi_a <= phi_0 + c1 * alpha * dphi_0
}

armijo_oks <- function(step0, step_a, c1) {
  armijo_ok(step0$f, step0$d, step_a$alpha, step_a$f, c1)
}

#' Curvature condition
curvature_ok <- function(d0, da, c2) {
  da > c2 * d0
}

curvature_oks <- function(step0, step_a, c2) {
#  message("step0$d ", formatC(step0$d), " stepa$d ", formatC(step_a$d), " c2 = ", c2)
  curvature_ok(step0$d, step_a$d, c2)
}

strong_curvature_ok <- function(d0, da, c2) {
  abs(da) <= -c2 * d0
}

strong_wolfe_ok <- function(f0, d0, alpha, fa, da, c1, c2) {
  armijo_ok(f0, d0, alpha, fa, c1) &&
    strong_curvature_ok(d0, da, c2)
}

strong_wolfe_oks <- function(step0, step, c1, c2) {
  armijo_ok(step0$f, step0$d, step$alpha, step$f, c1) &&
    strong_curvature_ok(step0$d, step$d, c2)
}


