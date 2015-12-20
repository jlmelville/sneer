# Optimization routines. Pure R. But it's probably the cost function and
# gradient evaluations that are the bottleneck.

#' Create optimizer.
#'
#' Function to create an optimization method. The optimizer consists of
#' \itemize{
#'  \item A method to calculate the gradient at a certain position.
#'  \item A method to calculate the direction to move in from the gradient
#'  position.
#'  \item A method to calculate the size of the step to move.
#'  \item A method to calculate the update of the solution, consisting of the
#'  gradient descent (as calculated by the previous three functions) and any
#'  extra modifiers, normally a momentum term based on the previous solution
#'  step.
#' }
#'
#' Normalizing the gradients (by setting \code{normalize_grads} to
#' \code{TRUE}) will scale the gradient matrix to length 1 before the
#' optimization step. This has the effect of making the size of the gradient
#' descent dependent only on the step size, rather than the product of the
#' step size and the size of the gradient. This can increase the stability of
#' step size methods like Bold Driver and the Jacobs method which iteratively
#' update the step size based on previous values rather than doing a search,
#' or with methods where the gradients can get extremely large (e.g. in
#' traditional distance-based embeddings like MDS and Sammon mapping which
#' involve dividing by potentially very small distances).
#'
#' The optimizer can validate the proposed solution, rejecting it
#' if the solution is not acceptable of the methods it is comprised of. It also
#' control updating the internal state of the methods (e.g. step size and
#' momentum).
#'
#' @param gradient Method to calculate the gradient at a solution
#' position. Set by calling a configuration function:
#' \itemize{
#'  \item \code{classical_gradient()} Calculates the gradient at the position of
#'  the current solution. This is the default setting.
#'  \item \code{nesterov_gradient()} Uses the Nesterov Accelerated Gradient
#'  method. A suitable \code{update} scheme should be used if this option is
#'  chosen, e.g. \code{nesterov_nsc_momentum}.
#' }
#' @param direction Method to calculate the direction to move. Set by calling a
#' configuration function:
#' \itemize{
#'  \item \code{steepest_descent()} Move in the direction of steepest descent.
#'  This is the default setting and it's the only currently defined method,
#'  so that's easy.
#' }
#' @param step_size Method to calculate the step size of the direction. Set
#' by calling a configuration function:
#' \itemize{
#'  \item \code{jacobs()} Jacobs method. Used in the t-SNE paper.
#'  \item \code{bold_driver()} Bold driver.
#' }
#' @param update Method to combine a gradient descent with other terms (e.g.
#' momentum) to produce the final update. Set by calling a configuration
#' function:
#' \itemize{
#'  \item \code{step_momentum()} Step momentum schedule. Used in the t-SNE paper.
#'  \item \code{linear_momentum()} Linear momentum schedule.
#'  \item \code{nesterov_nsc_momentum()} Nesterov momentum for non-strongly
#'  convex problems. Use when the \code{gradient} method is
#'  \code{nesterov_gradient}.
#'  \item \code{no_momentum()} Don't use a momentum term, optimization will
#'  only use gradient descent to update the solution. The default setting.
#' }
#' @param normalize_grads If \code{TRUE} the gradient matrix is normalized to
#' a length of one.
#' @param mat_name Name of the matrix in the output list \code{out} which
#' contains the embedded coordinates.
#' @param recenter If \code{TRUE}, recenter the coordinates after each
#' optimization step.
#' @return Optimizer.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding, and \code{\link{tsne_opt}} and \code{\link{bold_nag_opt}} for
#' some convenience functions that choose a set of defaults for you.
#' @examples
#' # Steepest descent with Jacobs adaptive step size and step momentum, as
#' # used in the t-SNE paper.
#' make_opt(step_size = jacobs(inc_fn = partial(`+`, 0.2), dec_mult = 0.8,
#'                             min_step_size = 0.1),
#'          update = step_momentum(initial_momentum = 0.5, final_momentum = 0.8,
#'                                 switch_iter = 250),
#'          normalize_grads = FALSE)
#'
#' # Use bold driver adaptive step size (1% step size increase, 25% decrease)
#' # with step momentum and normalizing gradients.
#' make_opt(step_size = bold_driver(inc_mult = 1.01, dec_mult = 0.75),
#'          update = step_momentum(initial_momentum = 0.5, final_momentum = 0.8,
#'                                 switch_iter = 250),
#'          normalize_grads = TRUE)
#'
#' # Nesterov Accelerated Gradient optimizer with bold driver adaptive step size
#' make_opt(gradient = nesterov_gradient(), step_size = bold_driver(),
#'          update = nesterov_nsc_momentum())
#'
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_sim(opt = make_opt(gradient = nesterov_gradient(),
#'                           step_size = bold_driver(),
#'                           update = nesterov_nsc_momentum()), ...)
#' }
make_opt <- function(gradient = classical_gradient(),
                     direction = steepest_descent(),
                     step_size = bold_driver(),
                     update = no_momentum(),
                     normalize_grads = TRUE, recenter = TRUE,
                     mat_name = "ym") {
  opt <- list()

  opt$mat_name <- mat_name
  opt$normalize_grads <- normalize_grads

  opt$grad_pos_fn <- gradient
  opt$direction_method <- direction
  opt$step_size_method <- step_size
  opt$update_method <- update

  da_fn <- NULL
  if (!is.null(direction$after_step)) {
    da_fn <- direction$after_step
  }
  sa_fn <- NULL
  if (!is.null(step_size$after_step)) {
    sa_fn <- step_size$after_step
  }
  ua_fn <- NULL
  if (!is.null(update$after_step)) {
    ua_fn <- update$after_step
  }

  opt$after_step <- make_after_step(direction_after_step_fn = da_fn,
                                    step_size_after_step_fn = sa_fn,
                                    update_after_step_fn = ua_fn,
                                    recenter = recenter)

  dv_fn <- NULL
  if (!is.null(direction$validate)) {
    dv_fn <- direction$validate
  }
  sv_fn <- NULL
  if (!is.null(step_size$validate)) {
    sv_fn <- step_size$validate
  }
  uv_fn <- NULL
  if (!is.null(update$validate)) {
    uv_fn <- update$validate
  }
  opt$validate <- make_validate_solution(direction_validation_fn = dv_fn,
                                         step_size_validation_fn = sv_fn,
                                         update_validation_fn = uv_fn)

  opt$init <- function(opt, inp, out, method) {
    if (!is.null(opt$direction_method$init)) {
      opt <- opt$direction_method$init(opt, inp, out, method)
    }
    if (!is.null(opt$step_size_method$init)) {
      opt <- opt$step_size_method$init(opt, inp, out, method)
    }
    if (!is.null(opt$update_method$init)) {
      opt <- opt$update_method$init(opt, inp, out, method)
    }

    opt
  }

  opt
}

#' t-SNE optimizer.
#'
#' Convenience factory function which sets the optimizer parameters to that
#' from the t-SNE paper.
#'
#' @return optimizer with parameters from the t-SNE paper.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding, and \code{\link{make_opt}} for a more generic function.
#' @examples
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_sim(opt = tsne_opt(), ...)
#' }
tsne_opt <- function() {
  make_opt(gradient = classical_gradient(),
           direction = steepest_descent(),
           step_size = tsne_jacobs(),
           update = step_momentum(),
           normalize_grads = FALSE, recenter = TRUE,
           mat_name = "ym")
}

#' Nesterov Accelerated Gradient optimizer with bold driver
#'
#' Convenience factory function which makes a very performant optimizer. Mixes
#' the NAG descent method and momentum for non-strongly convex problems
#' formulated by Sutkever et al., along with the bold driver method for step
#' size.
#'
#' @return Optimizer with NAG parameters and bold driver step size.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding, and \code{\link{make_opt}} for a more generic function.
#' @examples
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_sim(opt = bold_nag_opt(), ...)
#' }
bold_nag_opt <- function(min_step_size = sqrt(.Machine$double.eps),
                         initial_step_size = 1) {
  make_opt(gradient = nesterov_gradient(),
           step_size = bold_driver(min_step_size = min_step_size,
                                   initial_step_size = initial_step_size),
           update = nesterov_nsc_momentum())
}

#' Create callback to be invoked after the solution is updated.
#'
#' The direction, step size and update methods of the optimizer may all have
#' to update their internal state after the solution has been updated (e.g.
#' updating step sizes or momentum terms according to a schedule). This function
#' accumulates all the provided function into a single callback that the
#' optimizer will invoke whenever it updates the solution.
#'
#' @param recenter If \code{TRUE}, translate the embedded coordinates so they
#' are centered around the origin.
#' @param direction_after_step_fn Function provided by the direction method to
#' be invoked after the solution has been updated.
#' @param step_size_after_step_fn Function provided by the step size method to
#' be invoked after the solution has been updated.
#' @param update_after_step_fn Function provided by the update method to be
#' invoked after the solution has been updated.
#' @return Callback to be invoked by the optimizer after the solution has been
#' updated.
make_after_step <- function(recenter = TRUE,
                            direction_after_step_fn = NULL,
                            step_size_after_step_fn = NULL,
                            update_after_step_fn = NULL) {
  after_step <- list()

  if (!is.null(direction_after_step_fn)) {
    after_step$direction <- direction_after_step_fn
  }

  if (!is.null(step_size_after_step_fn)) {
    after_step$step_size <- step_size_after_step_fn
  }

  if (!is.null(update_after_step_fn)) {
    after_step$update <- update_after_step_fn
  }

  if (recenter) {
    after_step$recenter <- function(opt, inp, out, new_out, ok, iter) {
      vm <- new_out[[opt$mat_name]]
      vm <- sweep(vm, 2, colMeans(vm))  # subtract colMeans from each column
      new_out[[opt$mat_name]] <- vm

      list(new_out = new_out)
    }
  }

  function(opt, inp, out, new_out, ok, iter) {
    for (name in names(after_step)) {
      result <- after_step[[name]](opt, inp, out, new_out, ok, iter)
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$out)) {
        out <- result$out
      }
      if (!is.null(result$new_out)) {
        new_out <- result$new_out
      }
    }
    list(opt = opt, inp = inp, out = out, new_out = new_out)
  }
}

#' Create solution validation callback.
#'
#' Makes a callback to be used by the optimizer to validate a solution.
#'
#' The direction, step size or update part of the optimization step can "veto"
#' a particular solution if it's not to their liking. For example, the bold
#' driver step size method will flag an update as not ok if the cost increases.
#' If the solution does not validate, then it will not be applied to the
#' solution and the next iteration step takes place from the same position as
#' the current iteration. Therefore it's important that the state of the
#' optimizer also be changed if the validation has failed or the same failure
#' will occur again. For example, in the case of the bold driver, a failed
#' solution results in the step size being decreased.
#'
#' The callback, when invoked, will call the validation functions provided by
#' the different components of the optimizer. If any of them return \code{FALSE}
#' the the validation callback will indicate the step failed.
#'
#' @param direction_validation_fn Validation function for the direction method.
#' @param step_size_validation_fn Validation function for the step size method.
#' @param update_validation_fn Validation function for the update method.
#' @return A validation callback.
make_validate_solution <- function(direction_validation_fn = NULL,
                                   step_size_validation_fn = NULL,
                                   update_validation_fn = NULL) {
  validate_solution <- list()

  if (!is.null(direction_validation_fn)) {
    validate_solution$direction_validation_func <- direction_validation_fn
  }

  if (!is.null(step_size_validation_fn)) {
    validate_solution$step_size_validation_func <- step_size_validation_fn
  }

  if (!is.null(update_validation_fn)) {
    validate_solution$update_validation_func <- update_validation_fn
  }

  function(opt, inp, out, new_out, method) {
    all_good <- TRUE

    for (name in names(validate_solution)) {
      result <- validate_solution[[name]](opt, inp, out, new_out, method)
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$out)) {
        out <- result$out
      }
      if (!is.null(result$new_out)) {
        new_out <- result$new_out
      }
      if (!is.null(result$method)) {
        method <- result$method
      }
      if (!is.null(result$ok)) {
        if (!result$ok) {
          all_good <- FALSE
        }
      }
    }
    list(ok = all_good, opt = opt, inp = inp, out = out, new_out = new_out,
         method = method)
  }
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
nesterov_grad_pos <- function(opt, inp, out, method) {
  prev_update <- opt$update_method$update
  mu <- opt$update_method$momentum

  opt$update_method$update <- mu * prev_update
  new_out <- update_solution(opt, inp, out, method)

  gradient(inp, new_out, method, opt$mat_name)
}

#' Nesterov Accelerated Gradient.
#'
#' Configuration function for optimizer gradient calculation.
#'
#' @return NAG calculation method.
nesterov_gradient <- function() {
  nesterov_grad_pos
}

#' Classical Gradient.
#'
#' Configuration function for optimizer gradient calculation.
#'
#' @return Classical gradient calculation method.
classical_gradient <- function() {
  classical_grad_pos
}

#' Bold driver step size.
#'
#' This function configures the 'Bold Driver' method for step size selection.
#' If the costdecreases after an optimization step occurs, then the step
#' size will be increased (normally by a conservative amount). If the cost
#' increases, then the step size is decreased (normally by a more drastic
#' amount).
#'
#' @param inc_mult Multiplier of the current step size when the cost
#' decreases. Should be greater than one to increase the step size. This
#' parameter is ignored if \code{inc_fun} is supplied.
#' @param dec_mult Multiplier of the current step size when the cost
#' increases. Should be smaller than one to decrease the step size. This
#' parameter is ignored if \code{dec_fun} is supplied.
#' @param inc_fn Function to apply to the current step size when the cost
#' decreases. Should return a value greater than the current step size.
#' @param dec_fn Function to apply to the current step size when the cost
#' increases. Should return a value smaller than the current step size.
#' @param initial_step_size Step size to attempt on the first step of
#' optimization.
#' @param min_step_size Minimum step size.
#' @param max_step_size Maximum step size.
#' @return Step size method, to be used by the Optimizer. A list containing:
#' \itemize{
#'  \item \code{inc_fn} Function to invoke to increase the step size.
#'  \item \code{dec_fn} Function to invoke to decrease the step size.
#'  \item \code{initial_step_size} Initial step size.
#'  \item \code{min_step_size} Minimum step size.
#'  \item \code{max_step_size} Maximum step size.
#'  \item \code{init} Function to do any needed initialization.
#'  \item \code{get_step_size} Function to return the current step size.
#'  \item \code{validate} Function to validate whether the current step was
#'  successful or not.
#'  \item \code{after_step} Function to do any needed updating or internal state
#'  before the next optimization step.
#' }
bold_driver <- function(inc_mult = 1.1, dec_mult = 0.5,
                        inc_fn = partial(`*`, inc_mult),
                        dec_fn = partial(`*`, dec_mult),
                        initial_step_size = 1,
                        min_step_size = sqrt(.Machine$double.eps),
                        max_step_size = NULL) {
  list(
    inc_fn = inc_fn,
    dec_fn = dec_fn,
    initial_step_size = initial_step_size,
    min_step_size = min_step_size,
    max_step_size = max_step_size,
    init = function(opt, inp, out, method) {
      opt$step_size_method$old_cost <- method$cost_fn(inp, out)
      opt$step_size_method$step_size <- opt$step_size_method$initial_step_size
      opt
    },
    get_step_size = function(opt, inp, out, method) {
      opt$step_size_method$step_size
    },
    validate = function(opt, inp, out, new_out, method) {
      cost <- method$cost_fn(inp, new_out)
      ok <- cost < opt$step_size_method$old_cost

      opt$step_size_method$cost <- cost
      list(ok = ok, opt = opt)
    },
    after_step = function(opt, inp, out, new_out, ok, iter) {
      s_old <- opt$step_size_method$step_size
      if (ok) {
        s_new <- opt$step_size_method$inc_fn(opt$step_size_method$step_size)
      } else {
        s_new <- opt$step_size_method$dec_fn(opt$step_size_method$step_size)
        opt$step_size_method$cost <- opt$step_size_method$old_cost
      }
      ds <- s_new - s_old

      opt$step_size_method$step_size <- s_old + ds
      opt$step_size_method$step_size <- clamp(opt$step_size_method$step_size,
                                              opt$step_size_method$min_step_size)
      opt$step_size_method$old_cost <- opt$step_size_method$cost

      list(opt = opt)
    }
  )
}


#' Jacobs method step size selection.
#'
#' This function creates the Jacobs method for step size selection.
#'
#' Also known as the delta-bar-delta method. The method is described in:
#' R.A. Jacobs. Increased rates of convergence through learning rate adaptation.
#' Neural Networks, 1:295–307, 1988.
#' In this implementation, the sign of the gradient is compared to the sign of
#' the step size at the previous iteration (note that this includes any momentum
#' term). If the signs are the same, then the step size is increased. If the
#' signs differ, it is assumed that the minimum has been skipped over, and the
#' step size is decreased.
#'
#' There are two differences from the method described in the paper:
#' \enumerate{
#'  \item As originally described, increases in the step size are achieved by
#'  adding a fixed amount to current step size, while decreases occur by
#'  multiplying the step size by a positive value less than one. The default
#'  arguments here use multipliers for both increase and decrease.
#'  \item In the paper, the sign of the gradient is compared to a weighted
#'  average of gradients from several previous steps. In this implementation,
#'  we use only the value from the previous step.
#' }
#'
#' To use the settings as given in the t-SNE paper, see the \code{tsne_jacobs}
#' function.
#'
#' @param inc_mult Multiplier of the current step size when the cost
#' decreases. Should be greater than one to increase the step size. This
#' parameter is ignored if \code{inc_fun} is supplied.
#' @param dec_mult Multiplier of the current step size when the cost
#' increases. Should be smaller than one to decrease the step size. This
#' parameter is ignored if \code{dec_fun} is supplied.
#' @param inc_fn Function to apply to the current step size when the cost
#' decreases. Should return a value greater than the current step size.
#' @param dec_fn Function to apply to the current step size when the cost
#' increases. Should return a value smaller than the current step size.
#' @param initial_step_size Step size to attempt on the first step of
#' optimization.
#' @param min_step_size Minimum step size.
#' @param max_step_size Maximum step size.
#' @return Step size method, to be used by the Optimizer. A list containing:
#' \itemize{
#'  \item \code{inc_fn} Function to invoke to increase the step size.
#'  \item \code{dec_fn} Function to invoke to decrease the step size.
#'  \item \code{initial_step_size} Initial step size.
#'  \item \code{min_step_size} Minimum step size.
#'  \item \code{max_step_size} Maximum step size.
#'  \item \code{init} Function to do any needed initialization.
#'  \item \code{get_step_size} Function to return the current step size.
#'  \item \code{validate} Function to validate whether the current step was
#'  successful or not.
#'  \item \code{after_step} Function to do any needed updating or internal state
#'  before the next optimization step.
#' }
jacobs <- function(inc_mult = 1.1, dec_mult = 0.5,
                   inc_fn = partial(`*`, inc_mult),
                   dec_fn = partial(`*`, dec_mult),
                   init_step_size = 1, min_step_size = .Machine$double.eps,
                   max_step_size = NULL) {
  list(
    inc_fn = inc_fn,
    dec_fn = dec_fn,
    initial_step_size = init_step_size,
    min_step_size = min_step_size,
    max_step_size = max_step_size,
    init = function(opt, inp, out, method) {
      v <- out[[opt$mat_name]]
      opt$step_size_method$step_size <-
        matrix(opt$step_size_method$initial_step_size, nrow(v), ncol(v))
      opt
    },
    get_step_size = function(opt, inp, out, method) {

      gm <- opt$gm
      old_step_size <- opt$step_size_method$step_size
      inc_fn <- opt$step_size_method$inc_fn
      dec_fn <- opt$step_size_method$dec_fn
      old_update <- opt$update_method$update
      min_step_size <- opt$step_size_method$min_step_size

      new_step_size <- jacobs_step_size(gm, old_step_size,
                                        old_update, inc_fn, dec_fn)

      # clamp to min_gain to avoid negative learning rate
      new_step_size <- clamp(new_step_size, min_step_size)
    }
  )
}

#' Jacobs step size method using parameters from the t-SNE paper.
#' @return A step size method suitable for use with t-SNE.
tsne_jacobs <- function() {
  jacobs(inc_fn = partial(`+`, 0.2), dec_mult = 0.8,
                      min_step_size = 0.1)
}

#' Calculate a new step size matrix based on the sign of the gradient versus
#' that of the previous step.
#'
#' @details For each direction, the sign of the gradient is compared with that
#' of the update in the previous time step. If it's the same sign, increase the
#' step size; if the sign has changed, then it's assumed that the minimum was
#' missed and the current location brackets the minimum. In this case, the step
#' size is decreased.
#'
#' @param gm Gradient matrix.
#' @param step_size Step size for the previous iteration.
#' @param update Update_method matrix for the previous iteration.
#' @param inc_fn Function to apply to \code{step_size} to increase its elements.
#' @param dec_fn Function to apply to \code{step_size} to decrease its elements.
#' @return the new step size.
jacobs_step_size <- function(gm, step_size, update, inc_fn, dec_fn) {
  # the step size also includes the sign accounting for the descent so if
  # the old step is the opposite sign of the current gradient that implies
  # the old gradient had the same sign
  inc_fn(step_size) * abs(sign(gm) != sign(update)) +
    dec_fn(step_size) * abs(sign(gm) == sign(update))
}

#' Step momentum.
#'
#' Create an callback for the optimizer to use to update the embedding solution.
#' Update is in the form of a step momentum function.
#'
#' @param initial_momentum Momentum value for the first \code{switch_iter}
#' iterations.
#' @param final_momentum Momentum value after \code{switch_iter} iterations.
#' @param switch_iter Iteration number at which to switch from
#' \code{initial_momentum} to \code{final_momentum}.
#' @param verbose if \code{TRUE}, log info about the momentum.
#' @return A solution update method for use by the optimizer. A list consisting
#' of:
#' \itemize{
#'  \item \code{initial_momentum} Initial momentum.
#'  \item \code{final_momentum} Final momentum.
#'  \item \code{mom_switch_iter} Switch iteration.
#'  \item \code{init} Function to do any needed initialization.
#'  \item \code{get_update} Function to return the current update, which will
#'  be added to current solution matrix.
#'  \item \code{after_step} Function to do any needed updating or internal state
#'  before the next optimization step.
#' }
step_momentum <- function(initial_momentum = 0.5, final_momentum = 0.8,
                          switch_iter = 250, verbose = TRUE) {
  list(
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    mom_switch_iter = switch_iter,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$initial_momentum
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
#' \code{initial_momentum} to \code{final_momentum}.
#' @param initial_momentum Momentum value for the first \code{switch_iter}
#' iterations.
#' @param final_momentum Momentum value after \code{switch_iter} iterations.
#' @return A solution update method for use by the optimizer. A list consisting
#' of:
#' \itemize{
#'  \item \code{initial_momentum} Initial momentum.
#'  \item \code{final_momentum} Final momentum.
#'  \item \code{init} Function to do any needed initialization.
#'  \item \code{get_update} Function to return the current update, which will
#'  be added to current solution matrix.
#'  \item \code{after_step} Function to do any needed updating or internal state
#'  before the next optimization step.
#' }
linear_momentum <- function(max_iter, initial_momentum = 0,
                            final_momentum = 0.9) {
  list(
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$initial_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      mu_i <- opt$update_method$initial_momentum
      mu_f <- opt$update_method$final_momentum
      mu <- (mu_f - mu_i) / max_iter
      opt$update_method$momentum <- (mu * iter) + mu_i

      list(opt = opt)
    }
  )
}

#' Momentum schedule for non-strongly convex problems.
#'
#' Create an callback for the optimizer to use to update the embedding solution.
#' Update is in the form of a momentum schedule suggested in:
#'
#' Sutskever, I., Martens, J., Dahl, G. and Hinton, G. E.
#' On the importance of momentum and initialization in deep learning.
#' 30th International Conference on Machine Learning, Atlanta, USA, 2013.
#' JMLR: W&CP volume 28.
#'
#' which finds its origins in a publication by Nesterov. The version given by
#' Sustkever et al. is \eqn{1-\frac{3}{t+5}}
#'
#' @return A solution update method for use by the optimizer. A list consisting
#' of:
#' \itemize{
#'  \item \code{initial_momentum} Initial momentum.
#'  \item \code{init} Function to do any needed initialization.
#'  \item \code{get_update} Function to return the current update, which will
#'  be added to current solution matrix.
#'  \item \code{after_step} Function to do any needed updating or internal state
#'  before the next optimization step.
#' }
nesterov_nsc_momentum <- function() {
  list(
    initial_momentum = 0.5,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$initial_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      opt$update_method$momentum <- 1 - (3 / (iter + 5))
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
#' \itemize{
#'  \item \code{init} Function to do any needed initialization.
#'  \item \code{get_update} Function to return the current update, which will
#'  be added to current solution matrix.
#' }
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
#' Carries out a solution update using a momentum term in additon to the
#' gradient update.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Update matrix, consisting of gradient update and momentum term.
momentum_update <- function(opt, inp, out, method) {
  direction <- opt$direction_method$direction
  step_size <- opt$step_size_method$step_size
  prev_update <- opt$update_method$update
  mu <- opt$update_method$momentum
  grad_update <- step_size * direction

  (mu * prev_update) + ((1 - mu) * grad_update)
}

#' Steepest Descent direction.
#'
#' Creates a gradient descent direction method for use by the optimizer.
#' Uses the direction of steepest descent.
#'
#' @return Steepest Descent direction method. A list consisting of:
#' \itemize{
#'  \item get_direction Function invoked by the optimizer to find the direction
#'  to move in the gradient descent part of the solution update.
#' }
steepest_descent <- function() {
  list(
    get_direction = function(opt, inp, out, method, iter) {
      list(direction = -opt$gm)
    }
  )
}
