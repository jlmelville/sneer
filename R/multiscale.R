#' Initialize With Multiscale Perplexity
#'
#' An initialization method for creating input probabilities.
#'
#' This function calculates multiple input probability matrices, corresponding
#' to multiple perplexities, then uses the average of these matrices for the
#' final probability matrix.
#'
#' To avoid local minima, only the largest perplexity value is used initially,
#' with smaller perplexities being added over time. The perplexities used are
#' powers of 2, i.e. 2, 4, 8, 16, with the maximum (and initial) perplexity
#' given by the formula:
#'
#' \deqn{\lfloor{\log_{2}(N/4)}\rfloor}{floor(log2(N / 4)}
#'
#' where N is the number of observations in the data set.
#'
#' The output function used for the embedding also needs to be adapted for each
#' scale. Because the perplexity of the output can't be directly controlled,
#' a parameter which can control the 'width' of the similarity function should
#' be altered. For JSE, which uses an exponential weight function, the value
#' of the precision parameter, beta, is given by:
#'
#' \deqn{\beta = K^{-\frac{2}{P}}}{beta = K ^ (-2 / P)}
#'
#' where K is the perplexity and P is the output dimensionality (normally 2
#' for visualization purposes). The precision, beta, is then used in the
#' exponential weighting function:
#'
#' \deqn{W = \exp(-\beta D)}{W = exp(-beta * D)}
#'
#' where D is the output distance matrix and W is the resulting output weight
#' matrix.
#'
#' Like the input probability, these output weight matrices are converted to
#' individual probability matrices and then averaged to create the final
#' output probability matrix.
#'
#' If the parameter \code{multiscale_out_fn} is not provided, then the scheme
#' above is used to create output functions. Any function with a parameter
#' called \code{beta} can be used, so for example embedding methods which use
#' the \code{\link{exp_weight}} and \code{\link{heavy_tail_weight}} weighting
#' functions can be used with the default function. The signature of
#' \code{multiscale_out_fn} is:
#'
#' \code{multiscale_out_fn(out, method, perplexity)}
#'
#' where \code{out} is the current output data, \code{method} is the embedding
#' method, and \code{perplexity} is a perplexity value used in the multiscaling
#' of the input probability. This function will be called once for each
#' perplexity, and a weight function should be returned, which itself has the
#' signature:
#'
#' \code{weight_fn(D)}
#'
#' where D is the output distance matrix, and the return value is a weight
#' matrix.
#'
#' @param num_scale_iters Number of iterations for the perplexity of the input
#' probability to change from the largest to the smallest value.
#' @param multiscale_out_fn Factory function to create an output function
#'  appropriate for each scale. See the Details section.
#' @param input_weight_fn Weighting function for distances. It should have the
#'   signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#'   of squared distances and \code{beta} is a real-valued scalar parameter
#'   which will be varied as part of the search to produce the desired
#'   \code{perplexity}. The function should return a matrix of weights
#'   corresponding to the transformed squared distances passed in as arguments.
#' @param verbose If \code{TRUE} print message about tricks during the
#' embedding.
#' @return Input initializer for use by an embedding function.
#' @seealso \code{\link{embed_prob}} for how to use this function for
#' configuring an embedding.
#'
#' \code{\link{inp_step_perp}} also uses multiple
#' perplexity values, but replaces the old probability matrix with that of the
#' new perplexity at each step, rather than averaging.
#'
#' @examples
#' \dontrun{
#' # Should be passed to the tricks argument of an embedding function.
#' # Scale the perplexity over 20 iterations
#'  embed_prob(init_inp = inp_multiscale(num_scale_iters = 20), ...)
#' }
#' @references
#' Lee, J. A., Peluffo-Ordonez, D. H., & Verleysen, M. (2014).
#' Multiscale stochastic neighbor embedding: Towards parameter-free
#' dimensionality reduction. In ESANN.
#' @family sneer input initializers
#' @export
inp_multiscale <- function(num_scale_iters,
                           multiscale_out_fn = multiscale_exp,
                           input_weight_fn = exp_weight,
                           verbose = TRUE) {
  inp_prob(
    function(inp, method, opt, iter, out) {

      max_scales <- max(floor(log2(nrow(inp$dm) / 4)), 1)
      step_every <- max(round(num_scale_iters / max(max_scales - 1, 1)), 1)

      if (is.null(method$num_scales) ||
          (method$num_scales < max_scales
           && (iter == 0 || iter %% step_every == 0))) {

        if (iter == 0) {
          if (verbose) {
            message("Perplexity will be calculated over ", formatC(max_scales),
                    " scales, scaling every ", step_every, " iters")
          }
          method$num_scales <- 1
          method$scale_type <- "multi"
          method$multiscale_out_fn <- multiscale_out_fn
          method$orig_weight_fn <- method$weight_fn
        }
        else {
          method$num_scales <- method$num_scales + 1
        }
        perp <- 2 ^ (max_scales - method$num_scales + 1)
        if (verbose) {
          message("Iter: ", iter,
                  " setting perplexity to ", formatC(perp))
        }
        inp <- single_perplexity(inp, perplexity = perp,
                                 input_weight_fn = input_weight_fn,
                                 verbose = verbose)$inp

        # initialize or update the running total and mean of pms for each
        # perplexity
        if (is.null(inp$pm_sum)) {
          inp$pm_sum <- inp$pm
        }
        else {
          inp$pm_sum <- inp$pm_sum + inp$pm
          inp$pm <- inp$pm_sum / method$num_scales
        }
        if (verbose) {
          summarize(inp$pm, "msP")
        }
      }
      list(inp = inp, method = method)
    },
    init_only = FALSE
  )
}

# Factory function which creates an output weight function for a given input
# perplexity, in the context of multiscale embedding.
multiscale_exp <- function(out, method, perplexity) {
  out_dim <- ncol(out$ym)
  prec <- perplexity ^ (-2 / out_dim)

  weight_fn <- partial(method$orig_weight_fn, prec)
  attr(weight_fn, 'type') <- attr(method$orig_weight_fn, 'type')
  weight_fn
}
