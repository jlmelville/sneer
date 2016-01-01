# Optimization configuration.

#' Optimization Methods
#'
#' The available methods and wrappers that can be used to create optimization
#' routines in sneer.
#'
#' @examples
#' make_opt(step_size = bold_driver())
#' make_opt(step_size = jacobs())
#' @keywords internal
#' @name optimization_methods
#' @family sneer optimization methods
NULL

#' Optimizer
#'
#' Function to create an optimization method, used in an embedding function.
#'
#' The optimizer consists of
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
#'   position. Set by calling one of the configuration functions listed in
#'   \code{\link{optimization_gradient}}.
#' @param direction Method to calculate the direction to move. Set
#'   by calling one of the configuration functions listed in
#'   \code{\link{optimization_direction}}.
#' @param step_size Method to calculate the step size of the direction. Set
#'   by calling one of the configuration functions listed in
#'   \code{\link{optimization_step_size}}.
#' @param update Method to combine a gradient descent with other terms (e.g.
#'   momentum) to produce the final update. Set by calling one of the
#'   configuration functions listed in \code{\link{optimization_update}}.
#' @param normalize_grads If \code{TRUE} the gradient matrix is normalized to
#'   a length of one before step size calculation.
#' @param mat_name Name of the matrix in the output list \code{out} which
#'   contains the embedded coordinates.
#' @param recenter If \code{TRUE}, recenter the coordinates after each
#'   optimization step.
#' @return Optimizer.
#' @seealso \code{\link{embed_prob}} for how to use this function for configuring
#'   an embedding.
#' @examples
#' # Steepest descent with Jacobs adaptive step size and step momentum, as
#' # used in the t-SNE paper.
#' make_opt(step_size = jacobs(inc_fn = partial(`+`, 0.2), dec_mult = 0.8,
#'                             min_step_size = 0.1),
#'          update = step_momentum(init_momentum = 0.5, final_momentum = 0.8,
#'                                 switch_iter = 250),
#'          normalize_grads = FALSE)
#'
#' # Use bold driver adaptive step size (1% step size increase, 25% decrease)
#' # with step momentum and normalizing gradients.
#' make_opt(step_size = bold_driver(inc_mult = 1.01, dec_mult = 0.75),
#'          update = step_momentum(init_momentum = 0.5, final_momentum = 0.8,
#'                                 switch_iter = 250),
#'          normalize_grads = TRUE)
#'
#' # Nesterov Accelerated Gradient optimizer with bold driver adaptive step size
#' make_opt(gradient = nesterov_gradient(), step_size = bold_driver(),
#'          update = nesterov_nsc_momentum())
#'
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_prob(opt = make_opt(gradient = nesterov_gradient(),
#'                           step_size = bold_driver(),
#'                           update = nesterov_nsc_momentum()), ...)
#' }
#' @family sneer optimization methods
#' @export
make_opt <- function(gradient = classical_gradient(),
                     direction = steepest_descent(),
                     step_size = bold_driver(),
                     update = no_momentum(),
                     normalize_grads = TRUE, recenter = TRUE,
                     mat_name = "ym") {
  list(
    mat_name = mat_name,
    normalize_grads = normalize_grads,
    recenter = recenter,

    gradient = gradient,
    direction = direction,
    step_size = step_size,
    update = update,

    init = initialize_optimizer,
    validate = validate_solution,
    after_step = after_step,

    report = opt_report
  )
}

#' t-SNE Optimizer
#'
#' Optimizer factory function.
#'
#' Wrapper around \code{\link{make_opt}} which sets the optimizer parameters to
#' that from the t-SNE paper.
#'
#' @return optimizer with parameters from the t-SNE paper.
#' @seealso \code{\link{embed_prob}} and \code{\link{embed_dist}} for how to use
#'  this function for configuring an embedding.
#' @examples
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_prob(opt = tsne_opt(), ...)
#' }
#' @family sneer optimization methods
#' @export
tsne_opt <- function() {
  make_opt(gradient = classical_gradient(),
           direction = steepest_descent(),
           step_size = tsne_jacobs(),
           update = step_momentum(),
           normalize_grads = FALSE, recenter = TRUE,
           mat_name = "ym")
}

#' Nesterov Accelerated Gradient Optimizer with Bold Driver
#'
#' Optimizer factory function.
#'
#' Wrapper around \code{\link{make_opt}} which makes a very performant
#' optimizer. Mixes the NAG descent method and momentum for non-strongly convex
#' problems formulated by Sutkever et al., along with the bold driver method
#' for step size.
#'
#' @note This optimizer is prone to converge prematurely in the face of sudden
#' changes to the solution landscape, such as can happen when certain
#' \code{\link{tricks}} are applied. In these cases, substantially increasing
#' the \code{min_step_size} parameter so that the bold driver doesn't reduce
#' the step size is highly recommended.
#'
#' @param min_step_size Minimum step size allowed.
#' @param init_step_size Initial step size.
#' @param max_momentum Maximum value the momentum may take.
#' @return Optimizer with NAG parameters and bold driver step size.
#' @seealso \code{\link{embed_prob}} and \code{\link{embed_dist}} for how to use
#'  this function for configuring an embedding.
#' @examples
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_prob(opt = bold_nagger(), ...)
#' }
#' @family sneer optimization methods
#' @export
bold_nagger <- function(min_step_size = sqrt(.Machine$double.eps),
                         init_step_size = 1,
                         max_momentum = 1) {
  make_opt(gradient = nesterov_gradient(),
           step_size = bold_driver(min_step_size = min_step_size,
                                   init_step_size = init_step_size),
           update = nesterov_nsc_momentum(max_momentum = max_momentum))
}

#' Steepest Descent Optimizer with No Momentum
#'
#' Optimizer factory function.
#'
#' Wrapper around \code{\link{make_opt}} that creates a simple (some might say
#' boring) optimizer that only does steepest descent, without any momentum
#' term. The gradient is normalized to length 1 and the bold driver step
#' size method is used to adaptively select the step size.
#'
#' @return Pure gradient (steepest) descent optimizer.
#' @seealso \code{\link{embed_prob}} and \code{\link{embed_dist}} for how to use
#'  this function for configuring an embedding.
#' @examples
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_prob(opt = gradient_descent(), ...)
#' }
#' @family sneer optimization methods
#' @export
gradient_descent <- function() {
  make_opt(gradient = classical_gradient(), direction = steepest_descent(),
           step_size = bold_driver(min_step_size = 0.01),
           update = no_momentum(),
           normalize_grads = TRUE)
}

#' Optimizer Initialization.
#'
#' The direction, step size and update components of the optimizer may all
#' provide initialization methods to set their internal state before the first
#' optimization iteration This function will run those initialization functions
#' as part of the overall optimizer intialization step.
#'
#' @param opt Optimizer to initialize.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return a list containing:
#' \item{\code{opt}}{Optimizer with its components initialized.}
initialize_optimizer <- function(opt, inp, out, method) {
  init <- remove_nulls(list(
    direction = opt$direction$init,
    step_size = opt$step_size$init,
    update = opt$update$init
  ))

  for (name in names(init)) {
    opt <- init[[name]](opt, inp, out, method)
  }
  opt
}

#' Optimizer Solution Validation
#'
#' Validation function of an optimization iteration.
#'
#' The direction, step size or update part of the optimization step can "veto"
#' a particular solution if it's not to their liking. For example, the bold
#' driver step size method will flag an update as not ok if the cost increases.
#' If the solution does not validate, then it will not be applied to the
#' solution and the next iteration step takes place from the same position as
#' the current iteration.
#'
#' When invoked, this function will call the validation functions provided by
#' the different components of the optimizer. If any of them return \code{FALSE}
#' this function will also return \code{FALSE} (i.e. all parts of the optimizer
#' must agree that the proposed solution is a good one). The optimizer will
#' reject a solution which fails this validation function and start the next
#' iteration with the same solution as it started the current iteration with.
#' To avoid the optimizer getting permanently stuck, any part of the optimizer
#' that can fail the validation should also update its internal state so that
#' a different update will be attempted on the next iteration.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data from the start of the iteration.
#' @param proposed_out Proposed updated output for this iteration.
#' @param method Embedding method.
#' @return A list containing:
#' \item{\code{opt}}{Optimizer.}
#' \item{\code{inp}}{Input data.}
#' \item{\code{out}}{Output data from the start of the iteration.}
#' \item{\code{proposed_out}}{Proposed updated output for this iteration.}
#' \item{\code{method}}{Embedding method.}
#' \item{\code{ok}}{Logical value, \code{TRUE} if \code{proposed_out} passed
#' validation, \code{FALSE} otherwise}
validate_solution <- function(opt, inp, out, proposed_out, method) {
  validate <- remove_nulls(list(
    direction = opt$direction$validate,
    step_size = opt$step_size$validate,
    update = opt$update$validate
  ))

  all_good <- TRUE

  for (name in names(validate)) {
    result <- validate[[name]](opt, inp, out, proposed_out, method)
    if (!is.null(result$opt)) {
      opt <- result$opt
    }
    if (!is.null(result$inp)) {
      inp <- result$inp
    }
    if (!is.null(result$out)) {
      out <- result$out
    }
    if (!is.null(result$proposed_out)) {
      proposed_out <- result$proposed_out
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
  list(ok = all_good, opt = opt, inp = inp, out = out,
       proposed_out = proposed_out, method = method)
}

#' Post Optimization Step
#'
#' A function for the optimizer to run after updating the solution on each
#' iteration. As part of this process it will run any \code{after_step}
#' functions provided by the direction, step size and update methods of the
#' optimizer.
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data from the start of the iteration.
#' @param new_out Output data which will be the starting solution for the next
#'   iteration of optimization. If the validation stage failed, then this may be
#'   the same solution as \code{out}.
#' @param ok \code{TRUE} if the current iteration passed validation,
#'   \code{FALSE} otherwise.
#' @param iter Current iteration number.
#' @return A list containing
#'   \item{\code{opt}}{Updated optimizer.}
#'   \item{\code{inp}}{Input data.}
#'   \item{\code{out}}{Output data from the start of the iteration.}
#'   \item{\code{new_out}}{New output to be used in the next iteration.}
after_step <- function(opt, inp, out, new_out, ok, iter) {
  after_step <- remove_nulls(list(
    direction = opt$direction$after_step,
    step_size = opt$step_size$after_step,
    update = opt$update$after_step
  ))

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

  if (opt$recenter) {
    new_out[[opt$mat_name]] <- scale(new_out[[opt$mat_name]], center = TRUE,
                                     scale = FALSE)
  }

  list(opt = opt, inp = inp, out = out, new_out = new_out)
}

#' One Step of Optimization
#'
#' @param opt Optimizer
#' @param method Embedding method.
#' @param inp Input data.
#' @param out Output data.
#' @param iter Iteration number.
#' @return List consisting of:
#'   \item{\code{opt}}{Updated optimizer.}
#'   \item{\code{inp}}{Updated input.}
#'   \item{\code{out}}{Updated output.}
optimize_step <- function(opt, method, inp, out, iter) {
  if (iter == 0) {
    opt <- opt$init(opt, inp, out, method)
  }

  grad_result <- opt$gradient$calculate(opt, inp, out, method)

  if (any(is.nan(grad_result$gm))) {
    stop("NaN in grad. descent at iter ", iter)
  }
  opt$gm <- grad_result$gm

  if (opt$normalize_grads) {
    opt$gm <- normalize(opt$gm)
  }

  direction_result <- opt$direction$calculate(opt, inp, out, method, iter)
  opt <- direction_result$opt

  step_size_result <- opt$step_size$calculate(opt, inp, out, method)
  opt <- step_size_result$opt

  update_result <- opt$update$calculate(opt, inp, out, method)
  opt <- update_result$opt

  proposed_out <- update_solution(opt, inp, out, method)

  # intercept whether we want to accept the new solution e.g. bold driver
  ok <- TRUE
  if (!is.null(opt$validate)) {
    validation_result <- opt$validate(opt, inp, out, proposed_out, method)
    opt <- validation_result$opt
    inp <- validation_result$inp
    out <- validation_result$out
    proposed_out <- validation_result$proposed_out
    method <- validation_result$method
    ok <- validation_result$ok
  }

  if (ok) {
    new_out <- proposed_out
  } else {
    new_out <- out
  }

  if (!is.null(opt$after_step)) {
    after_step_result <- opt$after_step(opt, inp, out, new_out, ok, iter)
    opt <- after_step_result$opt
    inp <- after_step_result$inp
    out <- after_step_result$out
    new_out <- after_step_result$new_out
  }

  list(opt = opt, inp = inp, out = new_out)
}

#' Output Data Update
#'
#' This function updates the embedded coordinates in the output data, based
#' on the update information in the Optimizer, as well as updating any
#' auxiliary output data that is dependent on the coordinates (e.g. distances
#' and probabilities)
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Updated \code{out}.
update_solution <- function(opt, inp, out, method) {
  new_out <- out
  new_solution <- new_out[[opt$mat_name]] + opt$update$value
  new_out[[opt$mat_name]] <- new_solution
  method$update_out_fn(inp, new_out, method)
}


#' Optimizer Diagnostics
#'
#' Simple diagnostics of the state of the optimization.
#'
#' @param opt Optimizer to report on.
#' @return a list containing (very) brief numeric summary of parts of the
#' optimizer:
#' \item{\code{grad_length}}{Length of the gradient.}
#' \item{\code{step_size}}{Value of the step size (or length of the step size
#'  vector).}
#' \item{\code{momentum}}{Momentum (if any).}
opt_report <- function(opt) {
  result <- list()
  if (!is.null(opt$gm)) {
    result$grad_length <- length_vec(opt$gm)
  }
  if (class(opt$step_size$value) == "matrix") {
    result$step_size <- length_vec(opt$step_size$value)
  }
  else {
    result$step_size <- opt$step_size$value
  }
  if (!is.null(opt$update$momentum)) {
    result$momentum <- opt$update$momentum
  }

  result
}
