#' Optimization Step Size Methods
#'
#' The available step size methods that can be used by the optimization routines
#' in sneer.
#'
#' @examples
#' make_opt(step_size = bold_driver())
#' make_opt(step_size = jacobs())
#' @keywords internal
#' @name optimization_step_size
#' @family sneer optimization step size methods
NULL

#' Fixed Step Size
#'
#' Factory function for creating an optimizer step size method.
#'
#' This function creates a step size method where the step size is always
#' a constant value.
#' @param step_size Value of the step_size.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # step size method:
#' make_opt(step_size = constant_step_size(0.1))
#' @export
#' @family sneer optimization step size methods
constant_step_size <- function(step_size = 1) {
  list(
    init = function(opt, inp, out, method) {
      opt$step_size$value <- step_size
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      list(opt = opt)
    }
  )
}

#' Bold Driver Step Size Method
#'
#' Factory function for creating an optimizer step size method.
#'
#' This function configures the 'Bold Driver' method for step size selection.
#' At each iteration, the initial step size tried is a (normally conservative)
#' increase of the step size at the previous step. If this step size does not
#' result in a decrease in the cost, the step size is progressively reduced
#' (normally by a more drastic amount than the increase) until the cost
#' does decrease (or the minimum step size is reached).
#'
#' This implementation is effectively a backtracking line search, but instead
#' of a Wolfe condition, a cost decrease is considered sufficient.
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
#' @param init_step_size Step size to attempt on the first step of
#' optimization.
#' @param min_step_size Minimum step size.
#' @param max_step_size Maximum step size.
#' @return Bold driver step size method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_step_size_interface}}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # step size method:
#' make_opt(step_size = bold_driver())
#' @export
#' @family sneer optimization step size methods
bold_driver <- function(inc_mult = 1.1, dec_mult = 0.5,
                        inc_fn = partial(`*`, inc_mult),
                        dec_fn = partial(`*`, dec_mult),
                        init_step_size = 1,
                        min_step_size = sqrt(.Machine$double.eps),
                        max_step_size = NULL) {
  list(
    inc_fn = inc_fn,
    dec_fn = dec_fn,
    init_step_size = init_step_size,
    min = min_step_size,
    max = max_step_size,
    init = function(opt, inp, out, method) {
      opt$step_size$value <- opt$step_size$init_step_size
      opt$step_size$ok <- TRUE
      opt
    },
    calculate = function(opt, inp, out, method, iter) {
      out0 <- opt$gradient$calculate_position(opt, inp, out, method, iter)
      y0 <- out0[[opt$mat_name]]

      out0 <- set_solution(inp, y0, method)
      cost0 <- calculate_cost(method, inp, out0)

      pm <- opt$direction$value

      step_size <- opt$step_size$value
      y_alpha <- y0 + (step_size * pm)
      out_alpha <- set_solution(inp, y_alpha, method)
      cost <- calculate_cost(method, inp, out_alpha)

      while (cost > cost0 && step_size > min_step_size) {
        step_size <- sclamp(opt$step_size$dec_fn(step_size),
                            min = opt$step_size$min,
                            max = opt$step_size$max)
        y_alpha <- y0 + (step_size * pm)
        out_alpha <- set_solution(inp, y_alpha, method)
        cost <- calculate_cost(method, inp, out_alpha)
      }
      opt$cost <- cost
      opt$step_size$value <- step_size
      list(opt = opt)
    },
    after_step = function(opt, inp, out, new_out, ok, iter) {
      s_old <- opt$step_size$value
      # increase the cost for the next step
      s_new <- opt$step_size$inc_fn(s_old)
      opt$step_size$value <- sclamp(s_new,
                                    min = opt$step_size$min,
                                    max = opt$step_size$max)

      list(opt = opt)
    }
  )
}

#' Jacobs Step Size Method
#'
#' Factory function for creating an optimizer step size method.
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
#' @param init_step_size Step size to attempt on the first step of
#' optimization.
#' @param min_step_size Minimum step size.
#' @param max_step_size Maximum step size.
#' @return Jacobs step size method, to be used by the optimizer.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_step_size_interface}}
#' for details on the functions and values defined for this method.
#' \code{\link{tsne_jacobs}} provides a wrapper around for this method to use
#' the settings as given in the t-SNE paper.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # step size method:
#' make_opt(step_size = jacobs())
#' @references
#' Jacobs, R. A. (1988).
#' Increased rates of convergence through learning rate adaptation.
#' \emph{Neural networks}, \emph{1}(4), 295-307.
#'
#' Janet, J. A., Scoggins, S. M., Schultz, S. M., Snyder, W. E., White, M. W.,
#' & Sutton, J. C. (1998, May).
#' Shocking: An approach to stabilize backprop training with greedy adaptive
#' learning rates.
#' In \emph{1998 IEEE International Joint Conference on Neural Networks Proceedings.}
#' (Vol. 3, pp. 2218-2223). IEEE.
#' @family sneer optimization step size methods
#' @export
jacobs <- function(inc_mult = 1.1, dec_mult = 0.5,
                   inc_fn = partial(`*`, inc_mult),
                   dec_fn = partial(`*`, dec_mult),
                   init_step_size = 1, min_step_size = .Machine$double.eps,
                   max_step_size = NULL) {
  list(
    inc_fn = inc_fn,
    dec_fn = dec_fn,
    init_step_size = init_step_size,
    min_step_size = min_step_size,
    max_step_size = max_step_size,
    init = function(opt, inp, out, method) {
      v <- out[[opt$mat_name]]
      opt$step_size$value <-
        matrix(opt$step_size$init_step_size, nrow(v), ncol(v))
      opt
    },
    calculate = function(opt, inp, out, method, iter) {

      gm <- opt$gm
      old_step_size <- opt$step_size$value
      inc_fn <- opt$step_size$inc_fn
      dec_fn <- opt$step_size$dec_fn
      old_update <- opt$update$value
      min_step_size <- opt$step_size$min_step_size

      new_step_size <- jacobs_step_size(gm, old_step_size,
                                        old_update, inc_fn, dec_fn)

      # clamp to min_gain to avoid negative learning rate
      opt$step_size$value <- clamp(new_step_size, min_step_size)
      list(opt = opt)
    }
  )
}

#' Jacobs Step Size Method from t-SNE Paper
#'
#' Factory function for creating an optimizer step size method.
#'
#' @return A step size method suitable for use with t-SNE.
#' @seealso The return value of this function is intended for internal use of
#' the sneer framework only. See \code{\link{optimization_step_size_interface}}
#' for details on the functions and values defined for this method.
#' @examples
#' # Use as part of the make_opt function for configuring an optimizer's
#' # step size method:
#' make_opt(step_size = tsne_jacobs())
#' @references
#' Van der Maaten, L., & Hinton, G. (2008).
#' Visualizing data using t-SNE.
#' \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
#'
#' Jacobs, R. A. (1988).
#' Increased rates of convergence through learning rate adaptation.
#' \emph{Neural networks}, \emph{1}(4), 295-307.
#'
#' @export
#' @family sneer optimization step size methods
tsne_jacobs <- function() {
  jacobs(inc_fn = partial(`+`, 0.2), dec_mult = 0.8,
         min_step_size = 0.1)
}

#' Jacobs Step Size Matrix Update
#'
#' Calculate a new step size matrix based on the sign of the gradient versus
#' that of the previous step.
#'
#' For each direction, the sign of the gradient is compared with that
#' of the update in the previous time step. If it's the same sign, increase the
#' step size; if the sign has changed, then it's assumed that the minimum was
#' missed and the current location brackets the minimum. In this case, the step
#' size is decreased.
#'
#' @param gm Gradient matrix.
#' @param step_size Step size for the previous iteration.
#' @param update update matrix for the previous iteration.
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
