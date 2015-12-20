# Optimization configuration.

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
