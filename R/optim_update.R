#' Optimization Update Methods
#'
#' The available update methods that can be used by the optimization routines
#' in sneer.
#'
#' @examples
#' make_opt(update = step_momentum(init_momentum = 0.4, final_momentum = 0.8))
#' make_opt(update = no_momentum())
#'
#' @keywords internal
#' @name optimization_update
#' @family sneer optimization update methods
NULL

#' Momentum Update Method
#'
#' Create an update method using a user-defined momentum scheme function.
#'
#' @param mu_fn Momentum function with signature \code{mu_fn(iter)} where
#'  \code{iter} is the iteration number and the function returns the momentum
#'  value for that iteration. Ignored if \code{calculate} is specified.
#' @param calculate Momentum function with signature
#'  \code{calculate(opt, inp, out, method, iter)} where and the function
#'  returns the momentum value for that iteration.
#' @param validate Validation function.
#' @param after_step After-step function.
#' @param linear_weight If \code{TRUE}, then the gradient descent contribution
#'  to the update will be weighted by \code{1 - momentum}.
#' @param init_momentum Initial momentum value.
#' @param min_momentum Minimum momentum value.
#' @param max_momentum Maximum momentum value.
#' @param momentum Momentum method list containing any specific values required
#'  by a particular scheme.
#' @param msg_fn Function with signature \code{msg_fn(iter)} where
#'  \code{iter} is the iteration number. Called on each iteration if
#'  \code{verbose} is \code{TRUE}. Can be used to provide information.
#' @param verbose If \code{TRUE}, then \code{msg_fn} will be invoked on each
#'  iteration.
#' @return Momentum scheme update method.
#' @export
momentum_scheme <- function(mu_fn = NULL,
                            calculate = NULL,
                            validate = NULL,
                            after_step = NULL,
                            linear_weight = FALSE,
                            init_momentum = 0,
                            min_momentum = 0,
                            max_momentum = 1,
                            momentum = list(),
                            msg_fn = NULL,
                            verbose = TRUE) {
  momentum$min <- min_momentum
  momentum$max <- max_momentum
  momentum$value <- init_momentum

  if (!is.null(mu_fn)) {
    momentum$calculate <- function(opt, inp, out, method, iter) {
      sclamp(mu_fn(opt$update$t),
             min = opt$update$momentum$min,
             max = opt$update$momentum$max)
    }
  } else {
    momentum$calculate <- function(opt, inp, out, method, iter) {
      sclamp(calculate(opt, inp, out, method, iter),
             min = opt$update$momentum$min,
             max = opt$update$momentum$max)
    }
  }

  list(
    init = function(opt, inp, out, method) {
      opt$update$value <- matrix(0,
                                 nrow(out[[opt$mat_name]]),
                                 ncol(out[[opt$mat_name]]))
      opt$update$previous <- opt$update$value
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      if (verbose && !is.null(msg_fn)) {
        msg_fn(iter)
      }
      opt$update$value <- momentum_update(opt, inp, out, method, iter,
                                          linear_weight = linear_weight)
      list(opt = opt)
    },
    validate = validate,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      if (ok) {
        opt$update$previous <- opt$update$value
        opt$update$dirty <- TRUE
        opt$update$t <- opt$update$t + 1
      }
      else {
        opt$update$dirty <- FALSE
        opt$update$value <- opt$update$previous
      }
      if (!is.null(after_step)) {
        return(after_step(opt, inp, out, new_out, ok, iter))
      }
      list(opt = opt)
    },
    t = 0,
    dirty = TRUE,
    momentum = momentum
  )
}

#' Step Momentum
#'
#' Factory function for creating an optimizer update method.
#'
#' Create a callback for the optimizer to use to update the embedding solution.
#' Update is in the form of a step momentum function.
#'
#' @param init_momentum Momentum value for the first \code{switch_iter}
#' iterations.
#' @param final_momentum Momentum value after \code{switch_iter} iterations.
#' @param switch_iter Iteration number at which to switch from
#'  \code{init_momentum} to \code{final_momentum}.
#' @param ... Base momentum parameters to pass to the
#'  \code{momentum_scheme} factory function.
#' @return Step momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{optimization_update_interface}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = step_momentum())
#' @family sneer optimization update methods
#' @export
step_momentum <- function(init_momentum = 0.5, final_momentum = 0.8,
                          switch_iter = 250,
                          ...) {
  msg_fn <- function(iter) {
    if (iter == switch_iter) {
      message("Switching to final momentum ", formatC(final_momentum),
              " at iter ", iter)
    }
  }

  mu_fn <- function(iter) {
    if (iter >= switch_iter) {
      return(final_momentum)
    }
    else {
      return(init_momentum)
    }
  }

  momentum_scheme(mu_fn, msg_fn = msg_fn, ...)
}

#' Linear Momentum
#'
#' Factory function for creating an optimizer update method.
#'
#' Create a callback for the optimizer to use to update the embedding solution.
#' Update is in the form of a linear momentum function.
#'
#' @param max_iter Number of iterations to scale the momentum over from
#' \code{init_momentum} to \code{final_momentum}.
#' @param init_momentum Momentum value for the first \code{switch_iter}
#' iterations.
#' @param final_momentum Momentum value after \code{switch_iter} iterations.
#' @param ... Base momentum parameters to pass to the
#'  \code{momentum_scheme} factory function.
#' @return Linear momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{optimization_update_interface}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = linear_momentum())
#' @family sneer optimization update methods
#' @export
linear_momentum <- function(max_iter, init_momentum = 0,
                            final_momentum = 0.9, ...) {
  mu_fn <- function(iter) {
    mu_i <- init_momentum
    mu_f <- final_momentum
    mu <- (mu_f - mu_i) / max_iter
    (mu * iter) + mu_i
  }
  momentum_scheme(mu_fn, ...)
}

#' Nesterov Momentum for Non-Strongly Convex Functions
#'
#' Factory function for creating an optimizer update method.
#'
#' Create a callback for the optimizer to use to update the embedding solution,
#' using momentum scheme suggested by Nesterov.
#'
#' Update is in the form of a momentum schedule of the form
#' \deqn{\mu_{t} = 1-\frac{3}{t+5}}{mu_t = 1-[3/(t+5)]}
#' where \eqn{t} is the iteration number.
#'
#' @param ... Base momentum parameters to pass to the
#'  \code{momentum_scheme} factory function.
#' @return Nesterov momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{optimization_update_interface}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = nesterov_nsc_momentum())
#' @references
#' Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).
#' On the importance of initialization and momentum in deep learning.
#' In \emph{Proceedings of the 30th international conference on machine learning (ICML-13)}
#' (pp. 1139-1147).
#'
#' Sutskever, I. (2013).
#' \emph{Training recurrent neural networks}
#' (Doctoral dissertation, University of Toronto).
#' @family sneer optimization update methods
#' @export
nesterov_nsc_momentum <- function(...) {
  mu_fn <- function(iter) {
    1 - (3 / (iter + 5))
  }
  momentum_scheme(mu_fn, ...)
}

#' Constant Momentum
#'
#' Factory function for creating an optimizer update method.
#'
#' Create a callback for the optimizer to use with a constant momentum term.
#' This may be useful in the final stages of an optimization for fine tuning
#' a solution.
#'
#' @param momentum Momentum value to use.
#' @param ... Base momentum parameters to pass to the
#'  \code{momentum_scheme} factory function.
#' @return Constant momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{optimization_update_interface}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = constant_momentum())
#' @family sneer optimization update methods
#' @export
constant_momentum <- function(momentum, ...) {
  mu_fn <- function(iter) {
    momentum
  }
  momentum_scheme(mu_fn, ...)
}

#' No Momentum Update
#'
#' Factory function for creating an optimizer update method.
#'
#' Create an callback for the optimizer to use that uses strict gradient
#' descent with no momentum term.
#'
#' @param ... Base momentum parameters to pass to the
#'  \code{momentum_scheme} factory function.
#' @return Zero-momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{optimization_update_interface}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = no_momentum())
#' @family sneer optimization update methods
#' @export
no_momentum <- function(...) {
  list(
    init = function(opt, inp, out, method) {
      opt$update$value <- matrix(0,
                                 nrow(out[[opt$mat_name]]),
                                 ncol(out[[opt$mat_name]]))
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      opt$update$value <- gradient_update_term(opt)
      list(opt = opt)
    })
}

#' Update Solution With Momentum
#'
#' Carries out a solution update using a momentum term in addition to the
#' gradient update.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @param iter Iteration number.
#' @param linear_weight If \code{TRUE}, then the gradient descent contribution
#'  to the update will be weighted by \code{1 - momentum}.
#' @return Update matrix, consisting of gradient update and momentum term.
momentum_update <- function(opt, inp, out, method, iter, linear_weight = FALSE) {

  grad_term <- gradient_update_term(opt)
  momentum_term <- momentum_update_term(opt, inp, out, method, iter)
  if (linear_weight) {
    grad_weight <- (1 - opt$update$momentum$calculate(opt, inp, out, method,
                                                      iter))
  }
  else {
    grad_weight <- 1
  }
  momentum_term + (grad_weight * grad_term)
}

#' Momentum term of an update
#'
#' Calculates the momentum term of an update.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @param iter Iteration number.
#' @return Momentum update.
momentum_update_term <- function(opt, inp, out, method, iter) {
  momentum <- opt$update$momentum$calculate(opt, inp, out, method, iter)
  momentum * opt$update$previous
}

#' Gradient term of an update
#'
#' Calculates the gradient term of an update.
#'
#' @param opt Optimizer.
#' @return Gradient update.
gradient_update_term <- function(opt) {
  direction <- opt$direction$value
  step_size <- opt$step_size$value
  step_size * direction
}

#' Adaptive Restart
#'
#' Wrapper to apply an adaptive restart to the update scheme for an optimizer.
#'
#' O'Donoghue and Candes suggested an adaptive restart scheme for Nesterov
#' Accelerated Gradient scheme, where the update is effectively restarted if
#' a certain criterion is met. This function uses a cost-based criterion: if
#' the cost increases compared to the last iteration, the restart is applied.
#' This is intended for use with Nesterov Accelerated Gradient methods, but
#' it can be applied to other optimizers (whether it will do any good is
#' another matter).
#'
#' When a restart is applied, the momentum is treated as if the optimization
#' had been restarted: the iteration number for the purposes of the momentum
#' calculation is set back to zero, and any previous solutions are forgotten.
#' Future momentum updates are calculated using an effective iteration number
#' relative to  the last time the reset was applied.
#'
#' For schemes where the momentum increases non-linearly over the course of the
#' optimization, resetting the effective iteration number back to zero may be too
#' conservative. Therefore, in the spirit of the bold driver adaptive step size
#' algorithm, the extent to which the effective iteration number is reduced can
#' be provided with the \code{dec_mult} parameter, which should take a value
#' between 0 and 1, where 0 means that that no reduction in iteration is
#' applied, and 1 (the default) means that the full reduction is applied.
#'
#' @param update A momentum-based update method.
#' @param dec_mult In the event of a reset, the effective iteration number
#'  will be multiplied by this value.
#' @param dec_fn Function to decrease the effective iteration number after a
#'  reset. Should have the signature \code{dec_fn(iter)} where \code{iter} is
#'  the iteration number. Function should return a new iteration number between
#'  \code{0} and \code{iter}. Optional: default behavior is to multiply
#'  \code{iter} by \code{dec_mult}.
#' @return Update method with adaptive restart behavior.
#' @references
#' O'Donoghue, B., & Candes, E. (2013).
#' Adaptive restart for accelerated gradient schemes.
#' \emph{Foundations of computational mathematics}, \emph{15}(3), 715-732.
#' @export
#' @examples
#' update <- nesterov_nsc_momentum()
#' # give it adaptive restart powers,
#' # jump back to half iteration number when resetting
#' update <- adaptive_restart(update, dec_mult = 0.5)
#' \dontrun{
#' # pass to optimizer creation in an embedder
#' embed_prob(opt = nag(update = update), ...)
#' }
adaptive_restart <- function(update, dec_mult  = 0,
                             dec_fn = partial(`*`, dec_mult)) {

  if (!is.null(update$after_step)) {
    update$old_after_step <- update$after_step
  }
  if (!is.null(update$validate)) {
    update$old_validate <- update$validate
  }

  update$validate <- function(opt, inp, out, proposed_out, method, iter) {
    old_ok <- TRUE
    if (!is.null(opt$update$old_validate)) {
      result <- opt$update$old_validate(opt, inp, out, proposed_out, method, iter)
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$proposed_out)) {
        proposed_out <- result$proposed_out
      }
      if (!is.null(result$ok)) {
        old_ok <- result$ok
      }
    }
    result <- cost_validate(opt, inp, out, proposed_out, method, iter)
    if (result$ok && !old_ok) {
      result$ok <- FALSE
    }
    result
  }

  update$after_step <- function(opt, inp, out, new_out, ok, iter) {
    if (!is.null(opt$update$old_after_step)) {
      result <- opt$update$old_after_step(opt, inp, out, new_out, ok, iter)
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$out)) {
        out <- result$out
      }
    }

    if (!opt$cost_ok) {
      opt$update$value <- matrix(0, nrow(out[[opt$mat_name]]),
                               ncol(out[[opt$mat_name]]))
      opt$update$t <- dec_fn(opt$update$t)
      opt$update$dirty <- TRUE
    }

    list(opt = opt, inp = inp, new_out = new_out)
  }
  update
}

