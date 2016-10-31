# Integration code with the rconjgrad package at
# https://github.com/jlmelville/rconjgrad

# Conjugate Gradient Optimizer
#
# Function to create a conjugate gradient optimizer. Use of this optimizer
# requires installing and loading the 'rconjgrad' project from
# https://github.com/jlmelville/rconjgrad
#
# @param line_search Type of line search to use: \code{"mt"} for the
#  the method of More-Thuente, and \code{"r"} of Rasmussen.
# @param batch_iter CG optimization will be run for this number of iterations.
# @param prplus If \code{TRUE} then the 'PR+' variant of the Polak-Ribiere
#  update will be used: when the beta scale factor used to calculate the
#  new direction is negative, the search direction will be reset to
#  steepest descent.
# @param ortho_restart If \code{TRUE}, then if successive conjugate gradient
#  directions are not sufficiently orthogonal, reset the search direction to
#  steepest descent.
# @param nu If the dot product of the old and new conjugate gradient direction
#  (normalized with respect to inner product of the new direction) exceeds
#  this value, then the two directions are considered non-orthogonal and the
#  search direction is reset to steepest descent. Only used if
#  \code{ortho_restart} is \code{TRUE}.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   \code{c1} and 1.
# @param inc_iter If \code{TRUE}, increment the iteration number of the
#  embedding by the number of iterations used in the CG optimization. This makes
#  it easier to swap optimizers in the embedding routine, but may not be
#  suitable if other parts of the embedding routine (e.g. multiscaling,
#  reporting) are triggered on iteration numbers that are "skipped" due to
#  taking place inside the CG optimization.
# @return Optimizer.
optim_rcg <- function(line_search = "r", batch_iter = 20, prplus = TRUE,
                      ortho_restart = FALSE, nu = 0.1, c1 = c2 / 2, c2 = 0.1,
                      inc_iter = FALSE) {
  if (!requireNamespace("rconjgrad",
                        quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("Using conjugate gradient optimizer requires 'rconjgrad' package")
  }
  list(
    mat_name = "ym",
    optimize_step = rcg_opt_step,
    line_search = line_search,
    gradient = classical_gradient(),
    batch_iter = batch_iter,
    prplus = prplus,
    ortho_restart = ortho_restart,
    nu = nu,
    inc_iter = inc_iter,
    c1 = c1,
    c2 = c2,
    nfn = 0,
    ngr = 0,
    nresets = 0
  )
}

# One Round of Optimization using the CG optimizer.
#
# @note This function requires installing and loading the 'rconjgrad' project from
# https://github.com/jlmelville/rconjgrad
#
# @param opt Optimizer
# @param method Embedding method.
# @param inp Input data.
# @param out Output data.
# @param iter Iteration number.
# @return List consisting of:
#   \item{\code{opt}}{Updated optimizer.}
#   \item{\code{inp}}{Updated input.}
#   \item{\code{out}}{Updated output.}
rcg_opt_step <- function(opt, method, inp, out, iter) {
  fr <- make_optim_f(opt, inp, method, iter)
  grr <- make_optim_g(opt, inp, method, iter)

  par <- mat_to_par(out$ym)

  if (!requireNamespace("rconjgrad",
                        quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("Using conjugate gradient optimizer requires 'rconjgrad' package")
  }
  result <- rconjgrad::conj_grad(par = par, fn = fr, gr = grr,
                      line_search = opt$line_search,
                      max_iter = opt$batch_iter,
                      prplus = opt$prplus,
                      ortho_restart = opt$ortho_restart,
                      nu = opt$nu, c1 = opt$c1, c2 = opt$c2)

  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  out <- par_to_out(result$par, opt, inp, method, nr)
  if (opt$inc_iter) {
    iter <- iter + opt$batch_iter - 1
  }
  opt$nfn <- opt$nfn + result$counts[1]
  opt$ngr <- opt$ngr + result$counts[2]
  opt$nresets <- opt$nresets <- result$nresets
  list(opt = opt, inp = inp, out = out, iter = iter)
}

# More Thuente Line Search
#
# Line search method.
#
# @note This function requires installing and loading the 'rconjgrad' project from
# https://github.com/jlmelville/rconjgrad
#
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   \code{c1} and 1.
# @param max_alpha_mult Maximum scale factor to use when guessing the initial
#  step size for the next iteration
# @param stop_at_min If \code{TRUE}, then if the step size ever reaches
# \code{min_step_size}, stop the optimization.
# @return step size information for best step size.
more_thuente_ls <- function(c1 = c2 / 2, c2 = 0.1,
                            max_alpha_mult = 10,
                            min_step_size = .Machine$double.eps,
                            stop_at_min = TRUE) {
  if (!requireNamespace("rconjgrad",
                        quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("Using More-Thuente line search requires 'rconjgrad' package")
  }
  rcg_line_search(rconjgrad::more_thuente(c1 = c1, c2 = c2),
                  max_alpha_mult = max_alpha_mult,
                  min_step_size = min_step_size, stop_at_min = stop_at_min)
}

# Rasmussen Line Search
#
# Line search method.
#
# @note This function requires installing and loading the 'rconjgrad' project from
# https://github.com/jlmelville/rconjgrad
#
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   \code{c1} and 1.
# @param int Interpolation constant. Prevents step size being too small.
# @param ext Extrapolation constant. Prevents step size extrapolation being
#   too large.
# @param max_alpha_mult Maximum scale factor to use when guessing the initial
#  step size for the next iteration
# @param stop_at_min If \code{TRUE}, then if the step size ever reaches
# \code{min_step_size}, stop the optimization.
# @return step size information for best step size.
rasmussen_ls <- function(c1 = c2 / 2, c2 = 0.1, int = 0.1, ext = 3.0,
                            max_alpha_mult = 10,
                            min_step_size = .Machine$double.eps,
                            stop_at_min = TRUE) {
  if (!requireNamespace("rconjgrad",
                        quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("Using Rasmussen line search requires 'rconjgrad' package")
  }
  rcg_line_search(rconjgrad::rasmussen(c1 = c1, c2 = c2, int = int, ext = ext),
                  max_alpha_mult = max_alpha_mult,
                  min_step_size = min_step_size, stop_at_min = stop_at_min)
}

# Wolfe Condition Step Size Factory Function
#
# @note This function requires installing and loading the 'rconjgrad' project from
# https://github.com/jlmelville/rconjgrad
#
# @param ls_fn Line search function.
# @param max_alpha_mult Maximum scale factor to use when guessing the initial
#  step size for the next iteration
# @param stop_at_min If \code{TRUE}, then if the step size ever reaches
# \code{min_step_size}, stop the optimization.
# @return step size information for best step size.
rcg_line_search <- function(ls_fn,
                            max_alpha_mult = 10,
                            min_step_size = .Machine$double.eps,
                            stop_at_min = TRUE) {
  list(
    calculate = function(opt, inp, out, method, iter) {
      pm <- opt$direction$value
      phi_alpha <- make_phi_alpha(opt, inp, out, method, iter, pm,
                                  calc_gradient_default = TRUE)
      step0 <- phi_alpha(0)

      if (!is.null(opt$step_size$d0)) {
        slope_ratio <- opt$step_size$d0 / (step0$d + method$eps)
        opt$step_size$value <- opt$step_size$value *
          min(max_alpha_mult, slope_ratio)
      }
      else {
        opt$step_size$value <- 1 / (1 - step0$d)
      }

      ls_result <- ls_fn(phi_alpha, step0, opt$step_size$value)

      opt$step_size$d0 <- step0$d
      opt$step_size$value <- ls_result$step$alpha
      list(opt = opt)
    },
    validate = function(opt, inp, out, proposed_out, method, iter) {
      ok <- TRUE
      if (opt$step_size$value <= min_step_size && stop_at_min) {
        opt$stop_early <- TRUE
        ok <- FALSE
      }
      list(ok = ok, opt = opt)
    }
  )
}


