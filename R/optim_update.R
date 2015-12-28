#' Optimizer update methods.

#' Step momentum.
#'
#' Create an callback for the optimizer to use to update the embedding solution.
#' Update is in the form of a step momentum function.
#'
#' @param init_momentum Momentum value for the first \code{switch_iter}
#' iterations.
#' @param final_momentum Momentum value after \code{switch_iter} iterations.
#' @param switch_iter Iteration number at which to switch from
#' \code{init_momentum} to \code{final_momentum}.
#' @param verbose if \code{TRUE}, log info about the momentum.
#' @return A solution update method for use by the optimizer. A list consisting
#' of:
#'  \item{\code{init_momentum}}{Initial momentum.}
#'  \item{\code{final_momentum}}{Final momentum.}
#'  \item{\code{mom_switch_iter}}{Switch iteration.}
#'  \item{\code{init}}{Function to do any needed initialization.}
#'  \item{\code{get_update}}{Function to return the current update, which will
#'  be added to current solution matrix.}
#'  \item{\code{after_step}}{Function to do any needed updating or internal
#'  state before the next optimization step.}
#' @family sneer optimization update methods
#' @export
step_momentum <- function(init_momentum = 0.5, final_momentum = 0.8,
                          switch_iter = 250, verbose = TRUE) {
  list(
    init_momentum = init_momentum,
    final_momentum = final_momentum,
    mom_switch_iter = switch_iter,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$init_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      if (iter == opt$update_method$mom_switch_iter) {
        if (verbose) {
          message("Switching momentum to ", final_momentum, " at iter ", iter)
        }
        opt$update_method$momentum <- opt$update_method$final_momentum
      }

      list(opt = opt)
    }
  )
}

#' Linear momentum.
#'
#' Create an callback for the optimizer to use to update the embedding solution.
#' Update is in the form of a linear momentum function.
#'
#' @param max_iter Number of iterations to scale the momentum over from
#' \code{init_momentum} to \code{final_momentum}.
#' @param init_momentum Momentum value for the first \code{switch_iter}
#' iterations.
#' @param final_momentum Momentum value after \code{switch_iter} iterations.
#' @return A solution update method for use by the optimizer. A list consisting
#' of:
#'  \item{\code{init_momentum}}{Initial momentum.}
#'  \item{\code{final_momentum}}{Final momentum.}
#'  \item{\code{init}}{Function to do any needed initialization.}
#'  \item{\code{get_update}}{Function to return the current update, which will
#'  be added to current solution matrix.}
#'  \item{\code{after_step}}{Function to do any needed updating or internal state
#'  before the next optimization step.}
#' @family sneer optimization update methods
#' @export
linear_momentum <- function(max_iter, init_momentum = 0,
                            final_momentum = 0.9) {
  list(
    init_momentum = init_momentum,
    final_momentum = final_momentum,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$init_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      mu_i <- opt$update_method$init_momentum
      mu_f <- opt$update_method$final_momentum
      mu <- (mu_f - mu_i) / max_iter
      opt$update_method$momentum <- (mu * iter) + mu_i

      list(opt = opt)
    }
  )
}

#' Momentum schedule for non-strongly convex problems.
#'
#' Create an callback for the optimizer to use to update the embedding solution,
#' using momentum scheme suggested by Nesterov.
#'
#' Update is in the form of a momentum schedule of the form
#' \eqn{1-\frac{3}{t+5}}{1-(3/(t+5))}
#'
#' @param max_momentum Maximum value the momentum may take.
#' @return A solution update method for use by the optimizer. A list consisting
#' of:
#'  \item{\code{init_momentum}}{Initial momentum.}
#'  \item{\code{init}}{Function to do any needed initialization.}
#'  \item{\code{get_update}}{Function to return the current update, which will
#'  be added to current solution matrix.}
#'  \item{\code{after_step}}{Function to do any needed updating or internal state
#'  before the next optimization step.}
#' @references
#' Sutskever, I., Martens, J., Dahl, G. and Hinton, G. E.
#' On the importance of momentum and initialization in deep learning.
#' 30th International Conference on Machine Learning, Atlanta, USA, 2013.
#' JMLR: W&CP volume 28.
#' @family sneer optimization update methods
#' @export
nesterov_nsc_momentum <- function(max_momentum = 1) {
  list(
    init_momentum = 0.5,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$init_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      mu <- 1 - (3 / (iter + 5))
      opt$update_method$momentum <- min(max_momentum, mu)
      list(opt = opt)
    }
  )
}

#' An update schedule with no momentum.
#'
#' Create an callback for the optimizer to use that uses strict gradient
#' descent with no momentum term.
#'
#' @return A solution update method for use by the optimizer. A list consisting
#' of:
#'  \item{\code{init}}{Function to do any needed initialization.}
#'  \item{\code{get_update}}{Function to return the current update, which will
#'  be added to current solution matrix.}
#' @family sneer optimization update methods
#' @export
no_momentum <- function() {
  list(
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- 0
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update
  )
}

#' Solution update with momentum term.
#'
#' Carries out a solution update using a momentum term in addition to the
#' gradient update.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Update matrix, consisting of gradient update and momentum term.
momentum_update <- function(opt, inp, out, method) {
  direction <- opt$direction$value
  step_size <- opt$step_size$value
  prev_update <- opt$update_method$update
  mu <- opt$update_method$momentum
  grad_update <- step_size * direction

  (mu * prev_update) + ((1 - mu) * grad_update)
}
