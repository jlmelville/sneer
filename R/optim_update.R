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
#' \code{init_momentum} to \code{final_momentum}.
#' @param verbose if \code{TRUE}, log info about the momentum.
#' @return Step momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_update_interface}}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = step_momentum())
#' @family sneer optimization update methods
#' @export
step_momentum <- function(init_momentum = 0.5, final_momentum = 0.8,
                          switch_iter = 250, verbose = TRUE) {
  mu_fn <- function(iter) {
    if (iter == switch_iter) {
      return(final_momentum)
    }
    else {
      return(init_momentum)
    }
  }

  list(
    init = function(opt, inp, out, method) {
      opt$update$momentum <- mu_fn(0)
      opt$update$value <- matrix(0, nrow(out[[opt$mat_name]]),
                                    ncol(out[[opt$mat_name]]))
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      mu <- opt$update$mu_fn(iter)
      if (verbose && iter == switch_iter) {
        message("Switching to final momentum ", formatC(final_momentum),
                " at iter ", iter)
      }
      opt$update$value <- momentum_update(opt, inp, out, method, iter, mu)
      opt$update$momentum <- mu
      list(opt = opt)
    },
    mu_fn = mu_fn
  )
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
#' @return Linear momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_update_interface}}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = linear_momentum())
#' @family sneer optimization update methods
#' @export
linear_momentum <- function(max_iter, init_momentum = 0,
                            final_momentum = 0.9) {
  mu_fn <- function(iter) {
    mu_i <- init_momentum
    mu_f <- final_momentum
    mu <- (mu_f - mu_i) / max_iter
    (mu * iter) + mu_i
  }
  list(
    init = function(opt, inp, out, method) {
      opt$update$momentum <- mu_fn(0)
      opt$update$value <- matrix(0, nrow(out[[opt$mat_name]]),
                                    ncol(out[[opt$mat_name]]))
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      mu <- opt$update$mu_fn(iter)
      opt$update$value <- momentum_update(opt, inp, out, method, iter, mu)
      opt$update$momentum <- mu
      list(opt = opt)
    },
    mu_fn = mu_fn
  )
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
#' @param max_momentum Maximum value the momentum may take.
#' @return Nesterov momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_update_interface}}
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
#' @family sneer optimization update methods
#' @export
nesterov_nsc_momentum <- function(max_momentum = 1) {
  mu_fn <- function(iter) {
    mu <- 1 - (3 / (iter + 5))
    min(max_momentum, mu)
  }
  list(
    init = function(opt, inp, out, method) {
      opt$update$momentum <- mu_fn(0)
      opt$update$value <- matrix(0, nrow(out[[opt$mat_name]]),
                                    ncol(out[[opt$mat_name]]))
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      mu <- opt$update$mu_fn(iter)
      opt$update$value <- momentum_update(opt, inp, out, method, iter, mu)
      opt$update$momentum <- mu
      list(opt = opt)
    },
    mu_fn = mu_fn
  )
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
#' @return Constant momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_update_interface}}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = constant_momentum())
#' @family sneer optimization update methods
#' @export
constant_momentum <- function(momentum) {
  mu_fn <- function(iter) {
    momentum
  }
  list(
    init = function(opt, inp, out, method) {
      opt$update$momentum <- mu_fn(0)
      opt$update$value <- matrix(0, nrow(out[[opt$mat_name]]),
                                 ncol(out[[opt$mat_name]]))
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      mu <- opt$update$mu_fn(iter)
      opt$update$value <- momentum_update(opt, inp, out, method, iter, mu)
      opt$update$momentum <- mu
      list(opt = opt)
    },
    mu_fn = mu_fn
  )
}

#' No Momentum Update
#'
#' Factory function for creating an optimizer update method.
#'
#' Create an callback for the optimizer to use that uses strict gradient
#' descent with no momentum term.
#'
#' @return Zero-momentum update method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_update_interface}}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # update method:
#' make_opt(update = no_momentum())
#' @family sneer optimization update methods
#' @export
no_momentum <- function() {
  constant_momentum(0)
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
#' @param momentum Momentum for this iteration.
#' @return Update matrix, consisting of gradient update and momentum term.
momentum_update <- function(opt, inp, out, method, iter, momentum) {
  direction <- opt$direction$value
  step_size <- opt$step_size$value
  grad_update <- step_size * direction

  prev_update <- opt$update$value

  (momentum * prev_update) + grad_update
}
