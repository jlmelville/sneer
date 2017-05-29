# Initialize With Multiscale Perplexity
#
# An initialization method for creating input probabilities.
#
# This function calculates multiple input probability matrices, corresponding
# to multiple perplexities, then uses the average of these matrices for the
# final probability matrix.
#
# A list of perplexities may be provided to this function. Otherwise, the
# perplexities used are decreasing powers of 2, e.g. 16, 8, 4, 2
# with the maximum perplexity given by the formula:
#
# \deqn{\lfloor{\log_{2}(N/4)}\rceil}{round(log2(N / 4)}
#
# where N is the number of observations in the data set. The smallest
# perplexity value tried is 2.
#
# If a non-zero value of \code{num_scale_iters} is provided, the perplexities
# will be combined over the specified number of iterations, averaging over
# an increasing number of perplexities until they are all used to generate
# the probability matrix at \code{num_scale_iters}. If using the default
# scales, the perplexities are added in decreasing order, otherwise, they
# are added in the order provided in \code{scales} list. It is suggested that
# the \code{scales} list therefore order the perplexities in decreasing order.
#
# The output function used for the embedding also needs to be adapted for each
# scale. Because the perplexity of the output can't be directly controlled,
# a parameter which can control the 'width' of the similarity function should
# be altered. For JSE, which uses an exponential weight function, the value
# of the precision parameter, beta, is given by:
#
# \deqn{\beta = K^{-\frac{2U}{P}}}{beta = K ^ (-2U / P)}
#
# where K is the perplexity, P is the output dimensionality (normally 2
# for visualization purposes), and U is a coefficient which, when not equal to
# 1, represents a deviation from the assumption of a uniform density around
# each point (e.g. due to clustering, edge effects and so on). Empirically,
# Lee and co-workers suggested setting U to a value between 1 and 2, which they
# quantified as:
#
# \deqn{U = \textup{min}\left [2, \textup{max} \left( 1, \frac{\hat{D}}{P}\right ) \right ]}
# {U = min(2, max(1, D_hat / P))}
#
# and \eqn{\hat{D}}{D_hat} is the intrinsic dimensionality of the dataset.
#
# The intrinsic dimensionality of the dataset is the maximum
# intrinsic dimensionality calculated over a set of target perplexities. In
# turn, the intrinsic dimensionality for a given perplexity is given by
# averaging over the intrinsic dimensionality calculated at each point.
# The intrinsic dimensionality for a given data point and perplexity is
# calculated by:
#
# \deqn{\hat{D_{i,K}} = -2 \frac{\Delta H_{i,K}}{\Delta \log_{2}\beta_{i,K}}}
# {D_hat_ik = -2 * delta_h_ik / delta_log2b_ik}
#
# where \eqn{\frac{\Delta H}{\Delta \log_{2}\beta}}{delta_h_ik / delta_log2b_ik}
# is an estimate of the gradient of the Shannon Entropy (in bits) of the input
# probability i at perplexity K with respect to the log2 of the input
# precision. The estimate is made via a one-sided finite difference
# calculation, using the betas and perplexity values from the next largest
# perplexity in the \code{perplexities} vector.
#
# Yes, this seems like a lot of work to calculate a value between 1 and 2.
#
# The precision, beta, is then used in the exponential weighting function:
#
# \deqn{W = \exp(-\beta D)}{W = exp(-beta * D2)}
#
# where D2 is the output squared distance matrix and W is the resulting output
# weight matrix.
#
# Like the input probability, these output weight matrices are converted to
# individual probability matrices and then averaged to create the final
# output probability matrix.
#
# If the parameter \code{modify_kernel_fn} is not provided, then the scheme
# above is used to create output functions. Any function with a parameter
# called \code{beta} can be used, so for example embedding methods which use
# the \code{exp_weight} and \code{heavy_tail_weight} weighting
# functions can be used with the default function. The signature of
# \code{modify_kernel_fn} must be:
#
# \code{modify_kernel_fn(inp, out, method)}
#
# where \code{inp} is the input data, \code{out} is the current output data,
# \code{method} is the embedding method.
#
# This function will be called once for each perplexity, and an updated
# kernel should be returned.
#
# @param perplexities List of perplexities to use. If not provided, then
#   a series of perplexities in decreasing powers of two are used, starting
#   with the power of two closest to the number of observations in the dataset
#   divided by four.
# @param input_weight_fn Weighting function for distances. It should have the
#  signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#  of squared distances and \code{beta} is a real-valued scalar parameter
#  which will be varied as part of the search to produce the desired
#  perplexity. The function should return a matrix of weights
#  corresponding to the transformed squared distances passed in as arguments.
# @param num_scale_iters Number of iterations for the perplexity of the input
#  probability to change from the start perplexity to the end perplexity.
# @param modify_kernel_fn Function to create a new similarity kernel based
#  on the new perplexity. Will be called every time a new input probability
#  matrix is generated. See the details section for more.
# @param verbose If \code{TRUE} print message about initialization during the
# embedding.
# @return Input initializer for use by an embedding function.
# @seealso \code{embed_prob} for how to use this function for
# configuring an embedding.
#
# \code{inp_from_step_perp} also uses multiple
# perplexity values, but replaces the old probability matrix with that of the
# new perplexity at each step, rather than averaging.
#
# @examples
# \dontrun{
# # Should be passed to the init_inp argument of an embedding function.
# # Scale the perplexity over 20 iterations, using default perplexities
#  embed_prob(init_inp = inp_multiscale(num_scale_iters = 20), ...)
# # Scale the perplexity over 20 iterations using the provided perplexities
#  embed_prob(init_inp = inp_multiscale(num_scale_iters = 20,
#    scales = c(150, 100, 50, 25)), ...)
# # Scale using the provided perplexities, but use all of them at once
#  embed_prob(init_inp = inp_multiscale(num_scale_iters = 0,
#    scales = c(150, 100, 50, 25)), ...)
# }
# @references
# Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2014).
# Multiscale stochastic neighbor embedding: Towards parameter-free
# dimensionality reduction. In ESANN.
#
# Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
# Multi-scale similarities in stochastic neighbour embedding: Reducing
# dimensionality while preserving both local and global structure.
# \emph{Neurocomputing}, \emph{169}, 246-261.
# @family sneer input initializers
inp_from_perps_multi <- function(perplexities = NULL,
                                  input_weight_fn = exp_weight,
                                  num_scale_iters = NULL,
                                  modify_kernel_fn = scale_prec_to_perp,
                                  verbose = TRUE) {
  inp_prob(
    function(inp, method, opt, iter, out) {

      if (is.null(perplexities)) {
        perplexities <- ms_perps(inp$dm)
      }
      max_scales <- length(perplexities)
      method$max_scales <- max_scales
      method$perplexities <- perplexities

      if (is.null(num_scale_iters)) {
        num_scale_iters <- (max_scales - 1) * 100
      }
      num_steps <- max(max_scales - 1, 1)
      step_every <- num_scale_iters / num_steps

      if (iter == 0) {
        if (verbose) {
          message("Perplexity will be multi-scaled over ", max_scales,
                  " values, contributing every ~", round(step_every), " iters")
        }
        method$num_scales <- 0

        method <- ms_wrap_in_updated(method)

        method <- on_inp_updated(method, function(inp, out, method) {
          if (!is.null(modify_kernel_fn)) {
            if (!is.null(inp$pm)) {
              inp$pmtmp <- inp$pm
            }
            inp$pm <- inp$pms[[method$num_scales]]

            inp$beta <- inp$betas[[method$num_scales]]
            kernel <- modify_kernel_fn(inp, out, method)
          }
          else {
            kernel <- method$kernel
          }
          method$kernels[[method$num_scales]] <- kernel
          list(method = method)
        })$method

        method$orig_kernel <- method$kernel
        method$update_out_fn <- make_update_out_ms(
          unique(c(method$out_keep, "wm")))
        method$stiffness_fn <- plugin_stiffness_ms

        inp$pms <- list()
        inp$betas <- list()
        for (l in 1:max_scales) {
          perp <- perplexities[l]

          inpl <- single_perplexity(inp, perplexity = perp,
                                    input_weight_fn = input_weight_fn,
                                    verbose = verbose)$inp

          inp$pms[[l]] <- handle_prob(inpl$pm, method)
          inp$betas[[l]] <- inpl$beta
        }

        d_hat <- 1
        # Update intrinsic dimensionality using the method
        # suggested by Lee and co-workers
        for (l in 1:max_scales) {
          if (l != 1) {
            # need the next largest perplexity for finite difference estimation
            # so we can't do the calculation for the largest perplexity.

            # perplexities are stored in decreasing order, so the forward
            # finite difference estimate uses the values from the previous l
            beta_fwd <- inp$betas[[l - 1]]
            beta <- inp$betas[[l]]
            dlog2b <- log2(beta_fwd) - log2(beta)

            h_fwd <- log2(perplexities[l - 1])
            h <- log2(perplexities[l])
            dh <- h_fwd - h


            d_hat <- mean(-2 * dh / dlog2b)

            if (!is.null(inp$d_hat)) {
              inp$d_hat <- max(inp[["d_hat"]], d_hat)
            }
            else {
              inp$d_hat <- d_hat
            }
          }
        }
      }

      while (method$num_scales * step_every <= iter
             && method$num_scales < max_scales) {
        method$num_scales <- method$num_scales + 1

        inp$perp <- perplexities[method$num_scales]

        # initialize or update the running total and mean of
        # pms for each perplexity
        if (is.null(inp$pm_sum)) {
          inp$pm_sum <- inp$pms[[method$num_scales]]
        }
        else {
          inp$pm_sum <- inp$pm_sum + inp$pms[[method$num_scales]]
        }
        inp$pm <- inp$pm_sum / method$num_scales
        attr(inp$pm, 'type') <- attr(inp$pms[[method$num_scales]], 'type')

        if (verbose) {
          message("Iter ", iter, " scale ", method$num_scales, " adding perplexity ",
                  formatC(inp$perp), " to msP")
          summarize(inp$pm, "msP")
        }

        # because P can change more than once per iteration, we handle calling
        # inp_updated manually
        update_res <- inp_updated(inp, out, method)
        inp <- update_res$inp
        out <- update_res$out
        method <- update_res$method

        out$dirty <- TRUE
        opt$old_cost_dirty <- TRUE
        utils::flush.console()
      }
      list(inp = inp, method = method, out = out, opt = opt)
    },
    init_only = FALSE,
    call_inp_updated = FALSE
  )
}

# Initialize With Multiscale Perplexity (Lower Memory Usage)
#
# An initialization method for creating input probabilities.
#
# This function calculates multiple input probability matrices, corresponding
# to multiple perplexities, then uses the average of these matrices for the
# final probability matrix. It uses the same basic method of
# \code{inp_from_perps_multi}, but saves a bit of memory by not having
# to store all the probability matrices in memory at once.
#
# The requirement to store all the matrices in memory comes from needing
# to have calculated intrinsic dimensionalities at all scales to find the
# maximum value, which is used as the best estimate in calculating the degree
# to which the output precisions are modified to reflect any deviation from
# uniform density.
#
# If this wasn't necessary, it would only be necessary to store two matrices
# at any time: the running total of probability sums and the current
# probability matrix.
#
# The downside to this is that the estimate of the intrinsic dimensionality
# at a given iteration is only the largest value seen over the number of
# scales used in the average probability matrix so far, so that the output
# kernels for larger perplexities are slightly different compared to the
# full-memory version.
#
# The other difference between this version and the full-memory version is that
# because the perplexity/precision results for the next higher perplexity
# are not stored, they can't be used in the finite difference estimate of
# the dimensionality. Instead, values used in the initial bisection search
# to find the perplexity are used. This doesn't lead to very large differences
# in the intrinisic dimensionalities in various test cases (< 2%).
#
# A list of perplexities may be provided to this function. Otherwise, the
# perplexities used are decreasing powers of 2, e.g. 16, 8, 4, 2
# with the maximum perplexity given by the formula:
#
# \deqn{\lfloor{\log_{2}(N/4)}\rceil}{round(log2(N / 4)}
#
# where N is the number of observations in the data set. The smallest
# perplexity value tried is 2.
#
# If a non-zero value of \code{num_scale_iters} is provided, the perplexities
# will be combined over the specified number of iterations, averaging over
# an increasing number of perplexities until they are all used to generate
# the probability matrix at \code{num_scale_iters}. If using the default
# scales, the perplexities are added in decreasing order, otherwise, they
# are added in the order provided in \code{scales} list. It is suggested that
# the \code{scales} list therefore order the perplexities in decreasing order.
#
# The parameter \code{modify_kernel_fn} can be used to modify the output kernel
# based on the results of the perplexity calculation. If provided, then the
# signature of \code{modify_kernel_fn} must be:
#
# \code{modify_kernel_fn(inp, out, method)}
#
# where \code{inp} is the input data, \code{out} is the current output data,
# \code{method} is the embedding method.
#
# This function will be called once for each perplexity, and an updated
# kernel should be returned.
#
# @param perplexities List of perplexities to use. If not provided, then
#   a series of perplexities in decreasing powers of two are used, starting
#   with the power of two closest to the number of observations in the dataset
#   divided by four.
# @param input_weight_fn Weighting function for distances. It should have the
#  signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#  of squared distances and \code{beta} is a real-valued scalar parameter
#  which will be varied as part of the search to produce the desired
#  perplexity. The function should return a matrix of weights
#  corresponding to the transformed squared distances passed in as arguments.
# @param num_scale_iters Number of iterations for the perplexity of the input
#  probability to change from the start perplexity to the end perplexity.
# @param modify_kernel_fn Function to create a new similarity kernel based
#  on the new perplexity. Will be called every time a new input probability
#  matrix is generated. See the details section for more.
# @param verbose If \code{TRUE} print message about initialization during the
# embedding.
# @return Input initializer for use by an embedding function.
# @seealso
# \code{inp_from_perps_multi} for more details on how the intrinsic
# dimensionality is calculated.
# \code{embed_prob} for how to use this function for
# configuring an embedding.
inp_from_perps_multil <- function(perplexities = NULL,
                                 input_weight_fn = exp_weight,
                                 num_scale_iters = NULL,
                                 modify_kernel_fn = scale_prec_to_perp,
                                 verbose = TRUE) {
  inp_prob(
    function(inp, method, opt, iter, out) {

      if (is.null(perplexities)) {
        perplexities <- ms_perps(inp$dm)
      }

      max_scales <- length(perplexities)
      method$max_scales <- max_scales
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
        method <- on_inp_updated(method, function(inp, out, method) {
          if (!is.null(modify_kernel_fn)) {
            kernel <- modify_kernel_fn(inp, out, method)
          }
          else {
            kernel <- method$kernel
          }
          method$kernels[[method$num_scales]] <- kernel
          list(method = method)
        })$method

        method$orig_kernel <- method$kernel
        method$update_out_fn <- make_update_out_ms(
          unique(c(method$out_keep, "wm")))
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
        inp$pr <- inp$pm
        inp$pm <- handle_prob(inp$pm, method)

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

        # Update intrinsic dimensionality
        d_hat <- mean(inp$dims)
        if (!is.null(inp$d_hat)) {
          inp$d_hat <- max(inp$d_hat, d_hat)
        }
        else {
          inp$d_hat <- d_hat
        }

        # because P can change more than once per iteration, we handle calling
        # inp_updated manually
        update_res <- inp_updated(inp, out, method)
        inp <- update_res$inp
        out <- update_res$out
        method <- update_res$method

        out$dirty <- TRUE
        opt$old_cost_dirty <- TRUE
        utils::flush.console()
      }
      list(inp = inp, method = method, out = out)
    },
    init_only = FALSE,
    call_inp_updated = FALSE
  )
}

# Initialize With Step Perplexity
#
# An initialization method for creating input probabilities.
#
# This function initializes the input probabilities with a starting perplexity,
# then recalculates the input probability at different perplexity values for
# the first few iterations of the embedding. Normally, the embedding is begun
# at a relatively large perplexity and then the value is reduced to the
# usual target value over several iterations, recalculating the input
# probabilities. The idea is to avoid poor local minima. Rather than
# recalculate the input probabilities at each iteration by a linear decreasing
# ramp function, which would be time consuming, the perplexity is reduced
# in steps.
#
# You will need to decide what to do about the output function: should its
# parameters change as the input probabilities change? You could decide to
# do nothing, especially if you're using a kernel without any parameters, such
# as the Student t-distribution used in t-SNE, but you will need to explicitly
# set the \code{modify_kernel_fn} parameter to \code{NULL}.
#
# By default, the kernel function will try the  suggestion of Lee and
# co-workers, which is to scale the beta parameter of the exponential kernel
# function used in many embedding methods so that as the perplexity gets
# smaller, the beta value gets larger, thus reducing the bandwidth of the
# kernel. See the \code{scale_prec_to_perp} function for more details.
# If your kernel function doesn't have a \code{beta} parameter, the function
# will still run but have no effect on the output kernel.
#
# @param perplexities List of perplexities to use. If not provided, then
#   ten equally spaced perplexities will be used, starting at half the size
#   of the dataset, and ending at 32.
# @param input_weight_fn Weighting function for distances. It should have the
#  signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#  of squared distances and \code{beta} is a real-valued scalar parameter
#  which will be varied as part of the search to produce the desired
#  perplexity. The function should return a matrix of weights
#  corresponding to the transformed squared distances passed in as arguments.
# @param num_scale_iters Number of iterations for the perplexity of the input
#  probability to change from the start perplexity to the end perplexity.
# @param modify_kernel_fn Function to create a new similarity kernel based
#  on the new perplexity. Will be called every time a new input probability
#  matrix is generated. See the details section for more.
# @param verbose If \code{TRUE} print message about initialization during the
# embedding.
# @return Input initializer for use by an embedding function.
# @seealso \code{embed_prob} for how to use this function for
# configuring an embedding.
#
# @examples
# \dontrun{
# # Should be passed to the init_inp argument of an embedding function.
# # Step the perplexity from 75 to 25 with 6 values inclusive, taking 20
# # iterations overall (so 4 iterations per step)
#  embed_prob(init_inp = inp_from_step_perp(
#    perplexities = seq(75, 25, length.out = 6), num_scale_iters = 20), ...)
# }
# @references
# Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
# Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
# dimensionality reduction based on similarity preservation.
# \emph{Neurocomputing}, \emph{112}, 92-108.
#
# Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
# Information retrieval perspective to nonlinear dimensionality reduction for
# data visualization.
# \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#
# The paper by Venna and co-workers describes a very similar approach, but
# with decreasing the bandwidth of the input weighting function, rather than
# the perplexity. This is because NeRV sets the bandwidths of the output
# exponential similarity kernel to those from the input kernel.
#
# @family sneer input initializers
inp_from_step_perp <- function(perplexities = NULL,
                          input_weight_fn = exp_weight,
                          num_scale_iters = 20,
                          modify_kernel_fn = scale_prec_to_perp,
                          verbose = TRUE) {
  inp_prob(
    function(inp, method, opt, iter, out) {

      if (is.null(perplexities)) {
        perplexities <- step_perps(inp$dm)
      }

      max_scales <- length(perplexities)
      method$max_scales <- max_scales
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
          method <- on_inp_updated(method, function(inp, out, method) {
            method$kernel <- modify_kernel_fn(inp, out, method)
            list(method = method)
          })$method
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
        inp$pr <- inp$pm
        inp$pm <- handle_prob(inp$pm, method)

        # because P can change more than once per iteration, we handle calling
        # inp_updated manually
        update_res <- inp_updated(inp, out, method)
        inp <- update_res$inp
        out <- update_res$out
        method <- update_res$method

        out$dirty <- TRUE
        opt$old_cost_dirty <- TRUE
        utils::flush.console()

      }
      list(inp = inp, method = method, opt = opt)
    },
    init_only = FALSE,
    call_inp_updated = FALSE
  )
}

# Scale Output Kernel Precision Based on Intrinsic Dimensionality
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @family sneer kernel modifiers
scale_prec_to_perp <- function(inp, out, method) {

  # u differing from 1 allows deviation from assumption of strict uniformity
  # (e.g. due to clustering, edge effects) - clamping between 1 and 2 is
  # from an empirical observation made by Lee et al.
  u <- min(2, max(1, inp$d_hat / out$dim))

  # Lee et al (from whence this equation is taken) use prec/2 in their
  # exponential kernel, whereas the one in sneer uses prec directly.
  # just divide by 2 here to be consistent
  prec <- (inp$perp ^ ((-2 * u) / out$dim)) * 0.5
  if (method$verbose) {
    message("Creating kernel with precision ", formatC(prec),
            " for perplexity ", formatC(inp$perp),
            " d_intrinsic = ", formatC(inp$d_hat),
            " u = ", formatC(u))
  }
  new_kernel <- method$kernel
  new_kernel$beta <- prec
  new_kernel
}

# Initialize With Perplexity that Maximizes Intrinsic Dimensionality
#
# An initialization method for creating input probabilities.
#
# This technique uses a lot of the same ideas from multiscale embedding,
# except rather than use probabilities from multiple perplexities directly
# in the input and output spaces, it selects for the input probability matrix
# that which corresponds to the perplexity that gives the maximum intrinsic
# dimensionality.
#
# A list of perplexities may be provided to this function. Otherwise, the
# perplexities used are decreasing powers of 2, e.g. 16, 8, 4, 2
# with the maximum perplexity given by the formula:
#
# \deqn{\lfloor{\log_{2}(N/4)}\rceil}{round(log2(N / 4)}
#
# The perplexities are used in the order they are provided in the
# \code{perplexities} vector. As soon as the intrinsic dimensionality begins
# to decrease, the procedure is aborted and the probability matrix from the
# previously calculated perplexity is used.
#
# The parameter \code{modify_kernel_fn} can be used to modify the output kernel
# based on the results of the perplexity calculation. If provided, then the
# signature of \code{modify_kernel_fn} must be:
#
# \code{modify_kernel_fn(inp, out, method)}
#
# where \code{inp} is the input data, \code{out} is the current output data,
# \code{method} is the embedding method.
#
# This function will be called once for each perplexity, and an updated
# kernel should be returned.
#
# @param perplexities List of perplexities to use. If not provided, then
#   a series of perplexities in decreasing powers of two are used, starting
#   with the power of two closest to the number of observations in the dataset
#   divided by four.
# @param input_weight_fn Weighting function for distances. It should have the
#  signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#  of squared distances and \code{beta} is a real-valued scalar parameter
#  which will be varied as part of the search to produce the desired
#  perplexity. The function should return a matrix of weights
#  corresponding to the transformed squared distances passed in as arguments.
# @param modify_kernel_fn Function to create a new similarity kernel based
#  on the new perplexity. Will be called every time a new input probability
#  matrix is generated.
# @param verbose If \code{TRUE} print message about initialization during the
#  embedding.
# @return Input initializer for use by an embedding function.
# @seealso \code{inp_from_perps_multi} for caveats on using a
#  non-default version of \code{input_weight_fn}.
# @family sneer input initializers
inp_from_dint_max <- function(perplexities = NULL,
                              input_weight_fn = exp_weight,
                              modify_kernel_fn = NULL,
                              verbose = TRUE) {
  inp_prob(
    function(inp, method, opt, iter, out) {

      if (is.null(perplexities)) {
        perplexities <- ms_perps(inp$dm)
      }

      max_scales <- length(perplexities)
      method$max_scales <- max_scales
      method$perplexities <- perplexities

      if (!is.null(modify_kernel_fn)) {
        method <- on_inp_updated(method, function(inp, out, method) {
          method$kernel <- modify_kernel_fn(inp, out, method)
          list(method = method)
        })$method
      }
      method$orig_kernel <- method$kernel

      for (l in 1:max_scales) {
        perp <- perplexities[l]

        inpl <- single_perplexity(inp, perplexity = perp,
                                 input_weight_fn = input_weight_fn,
                                 verbose = verbose)$inp

        d_hat <- stats::median(inpl$dims)
        if (is.null(inp$d_hat)) {
          inp <- inpl
          inp$d_hat <- d_hat
          inp$perp <- perp
        }
        else {
          if (d_hat > inp$d_hat) {
            inp <- inpl
            inp$d_hat <- d_hat
            inp$perp <- perp
          }
          else {
            if (verbose) {
              message("Max intrinsic dim = ", formatC(inp$d_hat),
                      " found at perp ", formatC(perplexities[l - 1]))
            }
            break
          }
        }
      }
      list(inp = inp, method = method)
    })
}

# Multiscale Plugin Stiffness
#
# Calculates the stiffness matrix of an embedding method using the multiscale
# plugin gradient formulation.
#
# @param method Embedding method.
# @param inp Input data.
# @param out Output data.
# @return Stiffness matrix.
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

# Multiscale Plugin Stiffness for Row Probability
#
# Calculates the multiscale stiffness matrix for row probability based
# embedding methods.
#
# @param method Embedding method.
# @param inp Input data.
# @param out Output data.
# @return Stiffness matrix.
plugin_stiffness_ms_row <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)
  for (l in 1:method$num_scales) {
    dw_df <- method$kernels[[l]]$gr(method$kernels[[l]], out$d2m)

    wm_sum <- rowSums(out$wms[[l]])
    kml <- rowSums(dc_dq * out$qms[[l]]) # row sums
    kml <- sweep(dc_dq, 1, kml) # subtract row sum from each row element
    kml <- kml * (dw_df / (wm_sum + method$eps))
    kml <- kml + t(kml)

    if (l == 1) {
      kml_sum <- kml
    }
    else {
      kml_sum <- kml_sum + kml
    }
  }
  kml_sum / method$num_scales
}

# Multiscale Plugin Stiffness for Joint Probability
#
# Calculates the multiscale stiffness matrix for joint probability based
# embedding methods.
#
# @param method Embedding method.
# @param inp Input data.
# @param out Output data.
# @return Stiffness matrix.
plugin_stiffness_ms_cond <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)

  for (l in 1:method$num_scales) {
    kml <- plugin_stiffness_ms_pair(method, inp, out, dc_dq, l)

    kml <-  kml + t(kml)

    if (l == 1) {
      kml_sum <- kml
    }
    else {
      kml_sum <- kml_sum + kml
    }
  }

  kml_sum / method$num_scales
}

plugin_stiffness_ms_joint <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)

  for (l in 1:method$num_scales) {
    kml <- plugin_stiffness_ms_pair(method, inp, out, dc_dq, l)
    if (attr(method$kernels[[l]]$fn, 'type') == "symm") {
      kml <- 2 * kml
    }
    else {
      kml <- kml + t(kml)
    }

    if (l == 1) {
      kml_sum <- kml
    }
    else {
      kml_sum <- kml_sum + kml
    }
  }

  kml_sum / method$num_scales
}

# Shared plugin stiffness code for pair-wise normalization
plugin_stiffness_ms_pair <- function(method, inp, out, dc_dq, l) {
  dw_df <- method$kernels[[l]]$gr(method$kernels[[l]], out$d2m)
  wm_sum <- sum(out$wms[[l]])
  (dc_dq - sum(dc_dq * out$qms[[l]])) * (dw_df / (wm_sum + method$eps))
}


# Output Update Factory Function for Multiscale Probability
#
# In a multiscale embedding, the distance matrix is calculated once
# when the coordinates change, and then multiple similarity kernels
# are used to produce multiple weights matrices. These in turn
# create multiple probability matrices, which are averaged to create
# the final probability matrix.
#
# The weights and probabilities associated with the lth kernel is stored on
# the \code{wms[[l]]} and \code{qms[[l]]} list on \code{out}. The averaged
# probability matrix is stored as \code{out$qm} as usual.
#
# @return The output update function which will be invoked as part of the
# embedding.
make_update_out_ms <- function(keep = c("qm", "wm")) {
  function(inp, out, method) {
    out$d2m = coords_to_dist2(out$ym)

    out$qms <- list()
    out$wms <- list()

    for (l in 1:method$num_scales) {
      method$kernel <- method$kernels[[l]]
      res <- update_probs(out, method, d2m = out$d2m)

      for (i in 1:length(keep)) {
        keep_name <- paste0(keep[i], 's')
        out[[keep_name]][[l]] <- res[[keep[i]]]
      }
    }

    # average the probability matrices
    out$qm <- Reduce(`+`, out$qms) / length(out$qms)
    if (!is.null(method$out_updated_fn)) {
      out <- method$out_updated_fn(inp, out, method)
    }
    list(out = out, inp = inp)
  }
}

# Takes existing inp_updated listeners and wraps them so that they receive
# inp$pm and inp$beta from from the current scale
ms_wrap_in_updated <- function(method) {
  if (!is.null(method$num_inp_updated_fn)) {
    for (i in 1:method$num_inp_updated_fn) {
      if (!is.null(method$inp_updated_fns[[i]])) {
        unwrapped_fn <- method$inp_updated_fns[[i]]
        method$inp_updated_fns[[i]] <- function(inp, out, method) {
          inp$pm <- inp$pms[[method$num_scales]]
          inp$beta <- inp$betas[[method$num_scales]]
          update_result <- unwrapped_fn(inp, out, method)
          if (!is.null(update_result$inp)) {
            inp <- update_result$inp
          }
          if (!is.null(update_result$out)) {
            out <- update_result$out
          }
          if (!is.null(update_result$method)) {
            method <- update_result$method
          }
          list(inp = inp, out = out, method = method)
        }
      }
    }
  }
  method
}

# Perplexity Values for Multiscale Embedding
#
# Returns a vector of perplexity values scaled according to the size of the
# data frame or matrix suitable for use in a step wise scaling of perplexity.
#
# The perplexities will be returned in descending order, starting at the power
# of two closest to N/4, where N is the number of observations in the data set,
# and ending at 2, with the perplexities in taking up decreasing powers of 2.
#
# @param x Data frame or matrix.
# @param num_perps Number of perplexities.
# @return Vector containing \code{num_perps} perplexities.
ms_perps <- function(x, num_perps = max(round(log2(nrow(x) / 4)), 1)) {
  Filter(function(e) { e < nrow(x) },
         vapply(seq_along(1:num_perps),
                function(x) { 2 ^ (num_perps - x + 1) }, 0))
}

# Perplexity Values for Step Embedding
#
# Returns a vector of perplexity values scaled according to the size of the
# data frame or matrix suitable for use in a step wise scaling of perplexity.
#
# The perplexities will be returned in descending order, starting at N/2,
# where N is the number of observations in the data set, and ending at 32
# (or N/4, if there are less than 64 observations in the dataset), with
# the perplexities in between equally spaced.
#
# @param x Data frame or matrix.
# @param num_perps Number of perplexities.
# @return Vector containing \code{num_perps} perplexities.
step_perps <- function(x, num_perps = 5) {
  max_perp <- nrow(x) / 2
  min_perp <- 32
  if (max_perp <= min_perp || nrow(x) <= min_perp) {
    min_perp <- nrow(x) / 4
  }
  seq(max_perp, min_perp, length.out = num_perps)
}
