#' Tricks
#'
#' Miscellaneous data manipulations that may help optimize the embedding.
#' If you are fancy, you could call them "heuristics", but the t-SNE paper
#' calls their approach "tricks" and that's less characters to type, so we'll
#' go with that too.
#'
#' @examples
#' # pass one or more tricks to the make_tricks factory function:
#' # no tricks
#' make_tricks()
#'
#' # one trick
#' make_tricks(early_exaggeration())
#'
#' # two tricks
#' make_tricks(early_exaggeration(), late_momentum())
#'
#' \dontrun{
#' # in turn, pass the result of make_tricks to an embedding routine
#' embed_prob(make_tricks(early_exaggeration(), late_momentum()), ...)
#' }
#'
#' @keywords internal
#' @name tricks
#' @family sneer tricks
NULL


#' Tricks
#'
#' A collection of heuristics to improve embeddings.
#'
#' Creates a callback which the embedding routine will call before each
#' optimization step. Represents a collection of miscellaneous tweaks that
#' don't fit neatly into the rest of the framework, and as such can modify
#' the state of any other component of the embedding routine: input data,
#' output data, optimizer or embedding method.
#'
#' The input to this function can be zero, one or multiple independent tricks,
#' each of which is created by their own factory function. As a result, the
#' signature of this function is not informative. For information on the tricks
#' available, see \code{\link{tricks}}.
#'
#' Some wrappers around this function which apply sets of tricks from the
#' literature are available. See the links under the 'See Also' section.
#'
#' @param ... Zero or more \code{\link{tricks}}.
#' @return Callback collecting all the supplied tricks, to be invoked by the
#' embedding routine.
#' @seealso The result value of this function should be passed to the
#' \code{\link{tricks}} parameter of embedding routines like
#' \code{\link{embed_prob}} and \code{\link{embed_dist}}.
#' @examples
#' # Use early exaggeration as described in the t-SNE paper
#' make_tricks(early_exaggeration(exaggeration = 4, off_iter = 50))
#'
#' # Should be passed to the tricks argument of an embedding function:
#' \dontrun{
#'  embed_prob(tricks = make_tricks(early_exaggeration(
#'                                  exaggeration = 4, off_iter = 50), ...)
#' }
#' @family sneer trick collections
#' @export
make_tricks <- function(...) {
  tricks <- list(...)

  function(inp, out, method, opt, iter) {
    for (i in seq_along(tricks)) {
      result <- tricks[[i]](inp, out, method, opt, iter)
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$out)) {
        out <- result$out
      }
      if (!is.null(result$method)) {
        method <- result$method
      }
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
    }
    list(inp = inp, out = out, method = method, opt = opt)
  }
}

#' Early Exaggeration
#'
#' A "trick" for improving the quality of the embedding.
#'
#' This trick is for use with probability-based embedding. It scales up the
#' input probabilities for the first few iterations of the embedding to
#' encourage close distances to form between very similar points.
#'
#' @param exaggeration Size of the exaggeration factor: input probabilities will
#' be multiplied by this value.
#' @param off_iter Iteration step at which the input probabilities are returned
#' to their original values.
#' @param verbose If \code{TRUE} report a message when exaggeration is turned
#' off.
#' @return Trick callback. Should be passed to \code{\link{make_tricks}} when
#' configuring an embedding.
#' @examples
#' \dontrun{
#' # exaggerate for the first 100 iterations
#' embed_prob(make_tricks(early_exaggeration(off_iter = 100)), ...)
#' }
#' @family sneer tricks
#' @export
early_exaggeration <- function(exaggeration = 4, off_iter = 50,
                               verbose = TRUE) {
  function(inp, out, method, opt, iter) {
    if (iter == 0) {
      inp$pm <- inp$pm * exaggeration
    }
    if (iter == off_iter) {
      if (verbose) {
        message("Exaggeration off at iter: ", iter)
      }
      inp$pm <- inp$pm / exaggeration
    }

    list(inp = inp)
  }
}

#' Late Momentum
#'
#' A "trick" for improving the quality of the embedding.
#'
#' Replaces the momentum scheme of an optimizer with a fixed value. This may
#' be useful when using an optimizer with a very aggressive momentum schedule,
#' such as Nesterov Accelerated Gradient method. Reducing the momentum for the
#' final stage of the embedding may help the optimizer refine a solution.
#'
#' @param momentum Value of the fixed momentum to use.
#' @param on_iter Iteration step at which to apply the new momentum value.
#' @param verbose If \code{TRUE} report a message when the new momentum is
#' applied.
#' @return Trick callback. Should be passed to \code{\link{make_tricks}} when
#' configuring an embedding.
#' @examples
#' \dontrun{
#' # Apply new momentum after 800 iterations
#' embed_prob(make_tricks(late_exaggeration(momentum = 0.5, on_iter = 800)),
#'            ...)
#' }
#' @references
#' Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).
#' On the importance of initialization and momentum in deep learning.
#' In \emph{Proceedings of the 30th international conference on machine learning (ICML-13)}
#' (pp. 1139-1147).
#'
#' @family sneer tricks
#' @export
late_momentum <- function(momentum = 0.9, on_iter = 900, verbose = TRUE) {
  function(inp, out, method, opt, iter) {
    if (iter == on_iter) {
      if (!is.null(opt$update$momentum) && opt$update$momentum > momentum) {
        if (verbose) {
          message("Late momentum on at iter: ", iter)
        }
        opt$update <- constant_momentum(momentum)
        opt <- opt$update$init(opt, inp, out, method)
      }
    }
    list(opt = opt)
  }
}

#' t-SNE Tricks
#'
#' Tricks configured according to the details in the t-SNE paper.
#'
#' @param verbose If \code{TRUE} print message about tricks during the
#' embedding.
#' @return tricks callback parameterized to behave like the t-SNE paper.
#' @seealso \code{\link{embed_prob}} for how to use this function for
#' configuring an embedding.
#' @examples
#' # Should be passed to the tricks argument of an embedding function:
#' \dontrun{
#'  embed_prob(tricks = tsne_tricks(), ...)
#' }
#' @references
#' Van der Maaten, L., & Hinton, G. (2008).
#' Visualizing data using t-SNE.
#' \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
#' @family sneer trick collections
#' @export
tsne_tricks <- function(verbose = TRUE) {
  make_tricks(early_exaggeration(exaggeration = 4, off_iter = 50,
                                 verbose = verbose))
}

#' Step Perplexity
#'
#' A "trick" for improving the quality of the embedding.
#'
#' Recalculates the input probability at different perplexity values for the
#' first few iterations of the embedding. Normally, the embedding is begun
#' at a relatively large perplexity and then the value is reduced to the
#' usual target value over several iterations, recalculating the input
#' probabilities. The idea is to avoid poor local minima. Rather than
#' recalculate the input probabilities at each iteration by a linear decreasing
#' ramp function, which would be time consuming, the perplexity is reduced
#' in steps.
#'
#' @param start_perp Initial target perplexity value for the embedding.
#' @param stop_perp Final target perplexity value to be achieved after
#'  \code{num_iters} iterations.
#' @param num_iters Number of iterations for the perplexity of the input
#' probability to change from \code{start_perp} to \code{stop_perp}.
#' @param num_steps Number of discrete transitions of the perplexity to take
#' over \code{num_iters}. Cannot be larger than \code{num_iters}. The larger
#' this value, the smaller the change in the input probabilities when the
#' perplexity changes, but the more time spent in calculations.
#' @param input_weight_fn Weighting function for distances. It should have the
#'   signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#'   of squared distances and \code{beta} is a real-valued scalar parameter
#'   which will be varied as part of the search to produce the desired
#'   \code{perplexity}. The function should return a matrix of weights
#'   corresponding to the transformed squared distances passed in as arguments.
#' @param verbose If \code{TRUE} print message about tricks during the
#' embedding.
#' @return tricks callback to vary the perplexity.
#' @seealso \code{\link{embed_prob}} for how to use this function for
#' configuring an embedding. The paper by Venna and co-workers in the references
#' section describes a very similar method, but with decreasing the bandwidth
#' of the input weighting function, rather than the perplexity.
#' @examples
#' \dontrun{
#' # Should be passed to the tricks argument of an embedding function.
#' # Step the perplexity from 150 to 50 over 20 iterations, with 10 steps
#' # (so two iterations per step)
#'  embed_prob(tricks = step_perplexity(start_perp = 150, stop_perp = 50,
#'                                      num_iters = 20, num_steps = 10), ...)
#' }
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#' @family sneer trick collections
#' @export
step_perplexity <- function(start_perp, stop_perp, num_iters, num_steps = 10,
                            input_weight_fn = exp_weight, verbose = TRUE) {

  step_every <- max(round(num_iters / num_steps), 1)
  step_size <- (stop_perp - start_perp) / num_steps
  function(inp, out, method, opt, iter) {
    if (iter <= num_iters && (iter == 0 || iter %% step_every == 0)) {
      perp <- start_perp + ((iter / step_every) * step_size)
      if (verbose) {
        message("Iter: ", iter, " setting perplexity to ", formatC(perp))
      }
      inp <- inp_from_perp(perplexity = perp, input_weight_fn = input_weight_fn,
                           verbose = verbose)(inp, method)
      # invalidate cached data (e.g. old costs) in optimizer
      opt$cost_dirty <- TRUE
    }
    list(inp = inp, opt = opt)
  }
}
