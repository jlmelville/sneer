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
# Ensures that a step size isn't too long, by requiring the decrease
# in the function value to be sufficiently large compared to the step size
# (i.e. a step size that crosses the minimum and finished substantially up the
#  other side of the curve). The constant \param{c1} determines how much of
# a decrease is required, with a higher value requiring a larger decrease.
# According to Nocedal & Wright, it's normally set to a small value like 1e-4,
# which pretty much allows any function decrease.
#
# @param step0 Line function info at start point.
# @param step Line function info at a point.
# @param c1 Sufficient decrease constant. Must be between 0 and 1.
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
# Ensures that a step size isn't too short, by requiring the gradient of the
# line to be larger than a fraction of the starting gradient, i.e. assuming
# that you're moving in a descent direction (and you should be), the gradient
# should be less negative or even positive. In fact, this condition doesn't
# put a bound on how positive the gradient can get, so a step size which
# satisfies the criterion can be a long way from a minimizer.
#
# @param step0 Line function info at start point.
# @param step Line function info at a point along the line search.
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
  abs(da) <= c2 * abs(d0)
}

# Does Step Fulfil the Strong Curvature Rule
#
# Line Search Criterion.
#
# Ensures that a step size isn't too short, by requiring the absolute gradient
# of the line to be larger than a fraction of the absolute value of the starting
# gradient, i.e. assuming that you're moving in a descent direction (and you
# should be), the gradient should be less negative or slightly positive. Unlike
# the standard curvature condition, the strong curvature rule bounds how
# positive the gradient of the step size can be, so the value of the constant
# \param{c2} can be thought of as controlling how close the step size needs
# to be to the minimizer. Nocedal & Wright suggest c2 = 0.1 for a "tight"
# minimizer, and 0.9 for a looser search.
#
# @param step0 Line function info at start point.
# @param step Line function info at a point along the line search.
# @param c2 Curvature rule constant. Must take a value between 0 and 1.
# @return \code{TRUE} if \code{step_a} fulfils the strong curvature rule.
step_strong_curvature_ok <- function(step0, step, c2) {
  strong_curvature_ok(step0$d, step$d, c2)
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
# Tests that a step size lies close to a minimizer of a function, by using
# the Armijo (sufficient decrease) condition to test if the step size is too
# long, and the strong curvature condition to test if the step size is too
# short. The strength of the sufficient decrease condition is controlled by
# parameter \param{c1}, and the curvature condition by \param{c2}. \param{c1}
# should take a value between 0 and 1, and \param{c2} between \param{c1} and 1,
# i.e. 0 < c1 <= c2 < 1. Nocedal & Wright suggest always setting c1 to 1e-4,
# and c2 to 0.1 for a tight line search when getting close to the minimizer is
# important (e.g. for steepest descent and conjugate gradient), and to 0.9
# where a looser line search is sufficient (e.g quasi-Newton approaches like
# BFGS).
#
# @param step0 Line function info at start point.
# @param step Line function info at a point along the line search.
# @param c1 Sufficient decrease constant. Must be between 0 and 1.
# @param c2 Curvature decrease constant. Must be between \param{c1} and 1.
# @return \code{TRUE} if \code{step_a} fulfils the strong Wolfe conditions.
step_strong_wolfe_ok <- function(step0, step, c1, c2) {
  armijo_ok(step0$f, step0$d, step$alpha, step$f, c1) &&
    strong_curvature_ok(step0$d, step$d, c2)
}
