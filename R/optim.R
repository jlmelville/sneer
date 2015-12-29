# Optimization configuration.

#' Create optimizer.
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
#' position. Set by calling a configuration function:
#' \describe{
#'  \item{\code{\link{classical_gradient}}}{Calculates the gradient at the position of
#'  the current solution. This is the default setting.}
#'  \item{\code{\link{nesterov_gradient}}}{Uses the Nesterov Accelerated Gradient
#'  method. A suitable \code{update} scheme should be used if this option is
#'  chosen, e.g. \code{nesterov_nsc_momentum}}.
#' }
#' @param direction Method to calculate the direction to move. Set by calling a
#' configuration function:
#' \describe{
#'  \item{\code{\link{steepest_descent}}}{Move in the direction of steepest
#'  descent. This is the default setting and it's the only currently defined
#'  method, so that's easy.}
#' }
#' @param step_size Method to calculate the step size of the direction. Set
#' by calling a configuration function:
#' \describe{
#'  \item{\code{\link{jacobs}}}{Jacobs method. Used in the t-SNE paper.}
#'  \item{\code{\link{bold_driver}}}{Bold driver.}
#' }
#' @param update Method to combine a gradient descent with other terms (e.g.
#' momentum) to produce the final update. Set by calling a configuration
#' function:
#' \describe{
#'  \item{\code{\link{step_momentum}}}{Step momentum schedule. Used in the
#'  t-SNE paper.}
#'  \item{\code{\link{linear_momentum}}}{Linear momentum schedule.}
#'  \item{\code{\link{nesterov_nsc_momentum}}}{Nesterov momentum for
#'  non-strongly convex problems. Use when the \code{gradient} method is
#'  \code{\link{nesterov_gradient}}.}
#'  \item{\code{\link{no_momentum}}}{Don't use a momentum term, optimization
#'  will only use gradient descent to update the solution. The default setting.}
#' }
#' @param normalize_grads If \code{TRUE} the gradient matrix is normalized to
#' a length of one before step size calculation.
#' @param mat_name Name of the matrix in the output list \code{out} which
#' contains the embedded coordinates.
#' @param recenter If \code{TRUE}, recenter the coordinates after each
#' optimization step.
#' @return Optimizer.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding.
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
#'  embed_sim(opt = make_opt(gradient = nesterov_gradient(),
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
    after_step = after_step
  )
}

#' t-SNE optimizer.
#'
#' Convenience factory function which sets the optimizer parameters to that
#' from the t-SNE paper.
#'
#' @return optimizer with parameters from the t-SNE paper.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding.
#' @examples
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_sim(opt = tsne_opt(), ...)
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

#' Nesterov Accelerated Gradient optimizer with bold driver
#'
#' Convenience factory function which makes a very performant optimizer. Mixes
#' the NAG descent method and momentum for non-strongly convex problems
#' formulated by Sutkever et al., along with the bold driver method for step
#' size.
#' @param min_step_size Minimum step size allowed.
#' @param init_step_size Initial step size.
#' @param max_momentum Maximum value the momentum may take.
#' @return Optimizer with NAG parameters and bold driver step size.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding.
#' @examples
#' # Should be passed to the opt argument of an embedding function:
#' \dontrun{
#'  embed_sim(opt = bold_nag_opt(), ...)
#' }
#' @family sneer optimization methods
#' @export
bold_nag_opt <- function(min_step_size = sqrt(.Machine$double.eps),
                         init_step_size = 1,
                         max_momentum = 1) {
  make_opt(gradient = nesterov_gradient(),
           step_size = bold_driver(min_step_size = min_step_size,
                                   init_step_size = init_step_size),
           update = nesterov_nsc_momentum(max_momentum = max_momentum))
}

#' Optimizer initialization.
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

#' Validate proposed solution.
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
#' iteration of optimization. If the validation stage failed, then this may be
#' the same solution as \code{out}.
#' @param ok \code{TRUE} if the current iteration passed validation,
#' \code{FALSE} otherwise.
#' @param iter Current iteration number.
#' @return A list containing
#' \item{\code{opt}}{Updated optimizer.}
#' \item{\code{inp}}{Input data.}
#' \item{\code{out}}{Output data from the start of the iteration.}
#' \item{\code{new_out}}{New output to be used in the next iteration.}
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
