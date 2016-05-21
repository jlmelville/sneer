# Armijo Rule (or Sufficient Decrease Condition)
#
# @param phi_0 Value of phi at alpha = 0
# @param dphi_0 the directional derivative at alpha = 0
# @param alpha the step length.
# @param phi_a the value of phi at \code{alpha}.
# @param c1 the sufficient decrease constant.
# @return \code{TRUE} if the step \code{alpha} represents a sufficient decrease.
armijo_ok <- function(phi_0, dphi_0, alpha, phi_a, c1) {
  phi_a <= phi_0 + c1 * alpha * dphi_0
}

# Does Step Fulfil the Armijo Rule
#
# Line Search Criterion.
#
# @param step0 Line function info at start point.
# @param step Line function info at a point.
# @param c1 Sufficient decrease constant.
# @return \code{TRUE} if \code{step} fulfils the Armijo Rule.
step_armijo_ok <- function(step0, step, c1) {
  armijo_ok(step0$f, step0$d, step$alpha, step$f, c1)
}

# Curvature Condition
#
# @param d0 Value of gradient of the line at the starting position of the line
#  search.
# @param da Value of gradient of the line at a distance along the line.
# @param c2 Sufficient decrease constant.
# @return \code{TRUE} if the standard curvature condition is met.
curvature_ok <- function(d0, da, c2) {
  da > c2 * d0
}

# Does Step Fulfil the Standard Curvature Rule
#
# Line Search Criterion.
#
# @param step0 Line function info at start point.
# @param step Line function info at a point along the line searcht.
# @param c2 Curvature rule constant.
# @return \code{TRUE} if \code{step_a} fulfils the curvature rule.
step_curvature_ok <- function(step0, step, c2) {
  curvature_ok(step0$d, step$d, c2)
}

# Strong Curvature Condition
#
# @param d0 Value of gradient of the line at the starting position of the line
#  search.
# @param da Value of gradient of the line at a distance along the line.
# @param c2 Sufficient decrease constant.
# @return \code{TRUE} if the strong curvature condition is met.
strong_curvature_ok <- function(d0, da, c2) {
  abs(da) <= -c2 * d0
}

# Strong Wolfe Condition
#
# @param f0 Function value at the starting position of the line search.
# @param d0 Gradient value at the starting position of the line search.
# @param alpha Step length.
# @param fa Function value at distance \code{alpha} along the line.
# @param da Gradient value at distance \code{alpha} along the line.
# @param c1 Sufficient decrease constant.
# @param c2 Curvature decrease constant.
# @return \code{TRUE} if the strong Wolfe conditions are met.
strong_wolfe_ok <- function(f0, d0, alpha, fa, da, c1, c2) {
  armijo_ok(f0, d0, alpha, fa, c1) &&
    strong_curvature_ok(d0, da, c2)
}

# Does Step Fulfil the Strong Wolfe Condtions
#
# Line Search Criterion.
#
# @param step0 Line function info at start point.
# @param step Line function info at a point along the line searcht.
# @param c1 Sufficient decrease constant.
# @param c2 Curvature decrease constant.
# @return \code{TRUE} if \code{step_a} fulfils the strong Wolfe conditions.
step_strong_wolfe_ok <- function(step0, step, c1, c2) {
  armijo_ok(step0$f, step0$d, step$alpha, step$f, c1) &&
    strong_curvature_ok(step0$d, step$d, c2)
}
