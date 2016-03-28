#' Initialize With Multiscale Perplexity
#'
#' An initialization method for creating input probabilities.
#'
#' This function calculates multiple input probability matrices, corresponding
#' to multiple perplexities, then uses the average of these matrices for the
#' final probability matrix.
#'
#' A list of perplexities may be provided to this function. Otherwise, the
#' perplexities used are decreasing powers of 2, e.g. 16, 8, 4, 2
#' with the maximum perplexity given by the formula:
#'
#' \deqn{\lfloor{\log_{2}(N/4)}\rfloor}{floor(log2(N / 4)}
#'
#' where N is the number of observations in the data set. The smallest
#' perplexity value tried is 2.
#'
#' If a non-zero value of \code{num_scale_iters} is provided, the perplexities
#' will be combined over the specified number of iterations, averaging over
#' an increasing number of perplexities until they are all used to generate
#' the probability matrix at \code{num_scale_iters}. If using the default
#' scales, the perplexities are added in decreasing order, otherwise, they
#' are added in the order provided in \code{scales} list. It is suggested that
#' the \code{scales} list therefore order the perplexities in decreasing order.
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
#' of the input probability.
#'
#' This function will be called once for each perplexity, and an updated method
#' should be returned, with a new \code{weight_fn} function member with the
#' signature:
#'
#' \code{weight_fn(D)}
#'
#' where D is the output distance matrix, and the return value is a weight
#' matrix. The original weight function is stored as
#' \code{method$orig_weight_fn}.
#'
#' @param perplexities List of perplexities to use. If not provided, then
#'   ten equally spaced perplexities will be used, starting at half the size
#'   of the dataset, and ending at 32.
#' @param input_weight_fn Weighting function for distances. It should have the
#'  signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#'  of squared distances and \code{beta} is a real-valued scalar parameter
#'  which will be varied as part of the search to produce the desired
#'  perplexity. The function should return a matrix of weights
#'  corresponding to the transformed squared distances passed in as arguments.
#' @param num_scale_iters Number of iterations for the perplexity of the input
#'  probability to change from the start perplexity to the end perplexity.
#' @param modify_kernel_fn Function to create a new similarity kernel based
#'  on the new perplexity. Will be called every time a new input probability
#'  matrix is generated. See the details section for more.
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
#' # Should be passed to the init_inp argument of an embedding function.
#' # Scale the perplexity over 20 iterations, using default perplexities
#'  embed_prob(init_inp = inp_multiscale(num_scale_iters = 20), ...)
#' # Scale the perplexity over 20 iterations using the provided perplexities
#'  embed_prob(init_inp = inp_multiscale(num_scale_iters = 20,
#'    scales = c(150, 100, 50, 25)), ...)
#' # Scale using the provided perplexities, but use all of them at once
#'  embed_prob(init_inp = inp_multiscale(num_scale_iters = 0,
#'    scales = c(150, 100, 50, 25)), ...)
#' }
#' @references
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2014).
#' Multiscale stochastic neighbor embedding: Towards parameter-free
#' dimensionality reduction. In ESANN.
#'
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
#' @family sneer input initializers
#' @export
inp_from_perps_multi <- function(perplexities = NULL,
                                 input_weight_fn = exp_weight,
                                 num_scale_iters = NULL,
                                 modify_kernel_fn = scale_prec_to_perp,
                                 verbose = TRUE) {
  inp_prob(
    function(inp, method, opt, iter, out) {

      if (is.null(perplexities)) {
          max_scales <- max(round(log2(nrow(inp$dm) / 4)), 1)
          perplexities <- vapply(seq_along(1:max_scales),
                                 function(x) { 2 ^ (max_scales - x + 1) }, 0)
      }
      else {
        max_scales = length(perplexities)
      }
      method$perplexities <- perplexities

      if (is.null(num_scale_iters)) {
        num_scale_iters <- (max_scales - 1) * 100
      }
      num_steps <- max(max_scales - 1, 1)
      step_every <- num_scale_iters / num_steps

      if (iter == 0) {
        if (verbose) {
          message("Perplexity will be multi-scaled over ", max_scales,
                  " values, calculating every ~", round(step_every), " iters")
        }
        method$num_scales <- 0
        if (!is.null(modify_kernel_fn)) {
          method$inp_updated_fn <- function(inp, out, method) {
            kernel <- modify_kernel_fn(inp, out, method)
            method$kernels[[method$num_scales]] <- kernel
            list(method = method)
          }
        }
        method$orig_kernel <- method$kernel
        method$update_out_fn <- make_update_out_ms()
        method$stiffness_fn <- plugin_stiffness_ms
      }
      while (method$num_scales * step_every <= iter
             && method$num_scales < max_scales) {
        method$num_scales <- method$num_scales + 1

        perp <- perplexities[method$num_scales]

        if (verbose) {
          message("Iter: ", iter, " setting perplexity to ", formatC(perp))
        }


        inp <- single_perplexity(inp, perplexity = perp,
                                 input_weight_fn = input_weight_fn,
                                 verbose = verbose)$inp

        inp$perp <- perp


        # initialize or update the running total and mean of
        # pms for each perplexity
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


#' Initialize With Step Perplexity
#'
#' An initialization method for creating input probabilities.
#'
#' This function initializes the input probabilities with a starting perplexity,
#' then recalculates the input probability at different perplexity values for
#' the first few iterations of the embedding. Normally, the embedding is begun
#' at a relatively large perplexity and then the value is reduced to the
#' usual target value over several iterations, recalculating the input
#' probabilities. The idea is to avoid poor local minima. Rather than
#' recalculate the input probabilities at each iteration by a linear decreasing
#' ramp function, which would be time consuming, the perplexity is reduced
#' in steps.
#'
#' You will need to decide what to do about the output function: should its
#' parameters change as the input probabilities change? You could decide to
#' do nothing, especially if you're using a kernel without any parameters, such
#' as the Student t-distribution used in t-SNE, but you will need to explicitly
#' set the \code{modify_kernel_fn} parameter to \code{NULL}.
#'
#' By default, the kernel function will try the  suggestion of Lee and
#' co-workers, which is to scale the beta parameter of the exponential kernel
#' function used in many embedding methods so that as the perplexity gets
#' smaller, the beta value gets larger, thus reducing the bandwidth of the
#' kernel. See the \code{\link{scale_prec_to_perp}} function for more details.
#' If your kernel function doesn't have a \code{beta} parameter, the function
#' will still run but have no effect on the output kernel.
#'
#' @param perplexities List of perplexities to use. If not provided, then
#'   ten equally spaced perplexities will be used, starting at half the size
#'   of the dataset, and ending at 32.
#' @param input_weight_fn Weighting function for distances. It should have the
#'  signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#'  of squared distances and \code{beta} is a real-valued scalar parameter
#'  which will be varied as part of the search to produce the desired
#'  perplexity. The function should return a matrix of weights
#'  corresponding to the transformed squared distances passed in as arguments.
#' @param num_scale_iters Number of iterations for the perplexity of the input
#'  probability to change from the start perplexity to the end perplexity.
#' @param modify_kernel_fn Function to create a new similarity kernel based
#'  on the new perplexity. Will be called every time a new input probability
#'  matrix is generated. See the details section for more.
#' @param verbose If \code{TRUE} print message about tricks during the
#' embedding.
#' @return Input initializer for use by an embedding function.
#' @seealso \code{\link{embed_prob}} for how to use this function for
#' configuring an embedding.
#'
#' @examples
#' \dontrun{
#' # Should be passed to the init_inp argument of an embedding function.
#' # Step the perplexity from 75 to 25 with 6 values inclusive, taking 20
#' # iterations overall (so 4 iterations per step)
#'  embed_prob(init_inp = inp_step_perp(
#'    perplexities = seq(75, 25, length.out = 6), num_scale_iters = 20), ...)
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
#'
#' The paper by Venna and co-workers describes a very similar approach, but
#' with decreasing the bandwidth of the input weighting function, rather than
#' the perplexity. This is because NeRV sets the bandwidths of the output
#' exponential similarity kernel to those from the input kernel.
#'
#' @family sneer input initializers
#' @export
inp_step_perp <- function(perplexities = NULL,
                          input_weight_fn = exp_weight,
                          num_scale_iters = 20,
                          modify_kernel_fn = scale_prec_to_perp,
                          verbose = TRUE) {
  inp_prob(
    function(inp, method, opt, iter, out) {

      if (is.null(perplexities)) {
          max_scales <- 10
          max_perp <- nrow(out$ym) / 2
          min_perp <- 32
          if (nrow(out$ym) < min_perp) {
            min_perp <- 2
          }
          perplexities <- seq(max_perp, min_perp, length.out = max_scales)
      }
      else {
        max_scales = length(perplexities)
      }
      method$perplexities <- perplexities
      num_steps <- max(max_scales - 1, 1)
      step_every <- num_scale_iters / num_steps

      if (iter == 0) {
        if (verbose) {
          message("Perplexity will be single-scaled over ", max_scales,
                  " values, calculating every ~", round(step_every), " iters")
        }
        method$num_scales <- 0
        if (!is.null(modify_kernel_fn)) {
          method$inp_updated_fn <- function(inp, out, method) {
            method$kernel <- modify_kernel_fn(inp, out, method)
            list(method = method)
          }
        }
        method$orig_kernel <- method$kernel
      }

      while (method$num_scales * step_every <= iter
             && method$num_scales < max_scales) {
        method$num_scales <- method$num_scales + 1

        perp <- perplexities[method$num_scales]

        if (verbose) {
          message("Iter: ", iter,
                  " setting perplexity to ", formatC(perp))
        }

        inp <- single_perplexity(inp, perplexity = perp,
                                 input_weight_fn = input_weight_fn,
                                 verbose = verbose)$inp
        inp$perp <- perp
      }
      list(inp = inp, method = method)
    },
    init_only = FALSE
  )
}

#' Factory function which creates an output weight function for a given input
#' perplexity, in the context of multiscale embedding.
#'
#' @param inp Input data.
#' @param method Embedding method.
#' @param opt Optimizer.
#' @param iter Current iteration.
#' @param perp Perplexity of the input probability.
#' @param out Output data.
#' @param verbose If \code{TRUE} print message about tricks during the
#' embedding.
#' @family sneer kernel modifiers
scale_prec_to_perp <- function(inp, out, method) {
  # Lee et al (from whence this equation is taken) use prec/2 in their
  # exponential kernel, whereas the one in sneer uses prec directly.
  # just divide by 2 here to be consistent
  prec <- (inp$perp ^ (-2 / out$dim)) * 0.5
  if (method$verbose) {
    message("Creating kernel with precision ", formatC(prec),
            " for perplexity ", formatC(inp$perp))
  }
  new_kernel <- method$orig_kernel
  new_kernel$beta <- prec
  new_kernel
}

#' Multiscale Plugin Stiffness
#'
#' Calculates the stiffness matrix of an embedding method using the multiscale
#' plugin gradient formulation.
#'
#' @param method Embedding method.
#' @param inp Input data.
#' @param out Output data.
#' @return Stiffness matrix.
plugin_stiffness_ms <- function(method, inp, out) {
  prob_type <- method$prob_type
  if (is.null(prob_type)) {
    stop("Embedding method must have a prob type")
  }
  fn_name <- paste0('plugin_stiffness_ms_', prob_type)
  stiffness_fn <- get(fn_name)
  if (is.null(stiffness_fn)) {
    stop("Unable to find plugin stiffness function for ", prob_type)
  }
  stiffness_fn(method, inp, out)
}

#' Multiscale Plugin Stiffness for Row Probability
#'
#' Calculates the multiscale stiffness matrix for row probability based
#' embedding methods.
#'
#' @param method Embedding method.
#' @param inp Input data.
#' @param out Output data.
#' @return Stiffness matrix.
plugin_stiffness_ms_row <- function(method, inp, out) {
  cm_grad <- method$cost$gr(inp, out, method)

  for (l in 1:method$num_scales) {
    wm_sum <-  apply(out$wms[[l]], 1, sum)
    wm_grad <- method$kernels[[l]]$gr(method$kernels[[l]], out$d2m)
    kml <- apply(cm_grad * out$qms[[l]], 1, sum) # row sums
    kml <- sweep(-cm_grad, 1, -kml) # subtract row sum from each row element
    kml <- kml * (-wm_grad / wm_sum)
    kml <- 2 * (kml + t(kml))
    if (l == 1) {
      kml_sum <- kml
    }
    else {
      kml_sum <- kml_sum + kml
    }
  }

  kml_sum / method$num_scales
}

#' Multiscale Plugin Stiffness for Joint Probability
#'
#' Calculates the multiscale stiffness matrix for joint probability based
#' embedding methods.
#'
#' @param method Embedding method.
#' @param inp Input data.
#' @param out Output data.
#' @return Stiffness matrix.
plugin_stiffness_ms_joint <- function(method, inp, out) {
  cm_grad <- method$cost$gr(inp, out, method)

  for (l in 1:method$num_scales) {
    wm_sum <- sum(out$wms[[l]])
    wm_grad <- method$kernels[[l]]$gr(method$kernels[[l]], out$d2m)
    kml <- (sum(cm_grad * out$qms[[l]]) - cm_grad) * (-wm_grad / wm_sum)
    kml <- 2 * (kml + t(kml))
    if (l == 1) {
      kml_sum <- kml
    }
    else {
      kml_sum <- kml_sum + kml
    }
  }

  kml_sum / method$num_scales
}

#' Output Update Factory Function for Multiscale Probability
#'
#' In a multiscale embedding, the distance matrix is calculated once
#' when the coordinates change, and then multiple similarity kernels
#' are used to produce multiple weights matrices. These in turn
#' create multiple probability matrices, which are averaged to create
#' the final probability matrix.
#'
#' The weights and probabilities associated with the lth kernel is stored on
#' the \code{wms[[l]]} and \code{qms[[l]]} list on \code{out}. The averaged
#' probability matrix is stored as \code{out$qm} as usual.
#'
#' @return The output update function which will be invoked as part of the
#' embedding.
#' @export
make_update_out_ms <- function() {
  function(inp, out, method) {
    out$d2m = coords_to_dist2(out$ym)

    out$qms <- list()
    out$wms <- list()
    for (i in 1:method$num_scales) {
      method$kernel <- method$kernels[[i]]
      res <- update_probs(out, method, d2m = out$d2m)
      out$qms[[i]] <- res$qm
      out$wms[[i]] <- res$wm
    }

    # average the probability matrices
    out$qm <- Reduce(`+`, out$qms) / length(out$qms)

    if (!is.null(method$out_updated_fn)) {
      out <- method$out_updated_fn(inp, out, method)
    }
    out
  }
}
