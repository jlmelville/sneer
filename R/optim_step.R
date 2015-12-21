# Optimizer step size methods.

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
      opt$step_size_method$step_size <-
        clamp(opt$step_size_method$step_size,
              opt$step_size_method$min_step_size)
      opt$step_size_method$old_cost <- opt$step_size_method$cost

      list(opt = opt)
    }
  )
}


#' Jacobs method step size selection.
#'
#' This function creates the Jacobs method for step size selection. Also known
#' as the delta-bar-delta method.
#'
#' In this implementation, the sign of the gradient is compared to the sign of
#' the step size at the previous iteration (note that this includes any momentum
#' term). If the signs are the same, then the step size is increased. If the
#' signs differ, it is assumed that the minimum has been skipped over, and the
#' step size is decreased.
#'
#' There are two differences from the method as originally described by Jacobs:
#' \enumerate{
#'  \item As originally described, increases in the step size are achieved by
#'  adding a fixed amount to current step size, while decreases occur by
#'  multiplying the step size by a positive value less than one. The default
#'  arguments here use multipliers for both increase and decrease. See the
#'  paper by Janet and co-workers mentioned in the references for more details.
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
#' @references
#' R.A. Jacobs.
#' Increased rates of convergence through learning rate adaptation.
#' Neural Networks, 1:295–307, 1988.
#'
#' J.A. Janet, S.M. Scoggins, S.M. Schultz, W.E. Snyder, M.W. White,
#' J.C. Sutton III
#' Shocking: an approach to stabilize backprop training with greedy adaptive
#' learning rates
#' 1998 IEEE International Joint Conference on Neural Networks Proceedings.
#' IEEE World Congress on Computational Intelligence.
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
#'
#' @return A step size method suitable for use with t-SNE.
#' @references
#' Laurens van der Maarten, Geoffrey Hinton.
#' Visualizing Data using t-SNE.
#' Journal of Machine Learning Research, 2008, 9, 2579-2605.
#'
#' R.A. Jacobs.
#' Increased rates of convergence through learning rate adaptation.
#' Neural Networks, 1:295–307, 1988.
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
