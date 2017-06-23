# Functions to do the embedding.

# Probability-Based Embedding
#
# Carry out an embedding of a dataset using a probability-based method
# (e.g. t-SNE), with some useful default parameters.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{probability_embedding_methods}}.
# @param preprocess Input data preprocess callback. Set by assigning the
#   result value of \code{\link{make_preprocess}}.
# @param init_inp Input initializer. Set by assigning the result value of
#   calling one of the available \code{\link{input_initializers}}.
# @param init_out Output initializer. Set by assigning the result value of
#   calling one of the available \code{\link{output_initializers}}.
# @param opt Optimization method. Set by assigning the result of value of one
#   of the configuration functions listed in
#   \code{\link{optimization_methods}}.
# @param max_iter Maximum number of optimization steps to take.
# @param tricks Optional collection of heuristics. Set by assigning the result
#  value of \code{\link{make_tricks}} or a related wrapper. See
#  \code{\link{tricks}} for the available tricks.
# @param reporter Reporter callback. Set by assigning the result value of
#   \code{\link{make_reporter}}.
# @param export Vector of names, specifying data in addition to of the output
#   data to export provide in the return value. Possible names are:
#   \describe{
#     \item{"\code{inp}"}{The input data, based on the argument passed to
#       \code{init_inp}. See the help text for the specific
#       \code{\link{input_initializers}} for details.}
#     \item{"\code{report}"}{The result of the last report. See the help
#       text for \code{\link{make_reporter}} for details.}
#     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#     passed to \code{opt} for details.}
#   }
#   For each name supplied, an item with that name will appear in the return
#   value list.
# @param after_embed Callback to run on input and output data before output
#   data is returned.
# @param verbose If \code{TRUE} display messages about the embedding progress.
# @return The output data. A list containing:
#  \itemize{
#   \item \code{ym} Embedded coordinates.
#   \item \code{cost} The cost value associated with \code{ym}.
#   If the parameter \code{export} was used, additional elements will be
#   present. See the help text for the \code{export} parameter for more
#   details.
#   }
# @seealso
# \itemize{
# \item \code{\link{probability_embedding_methods}} for configuring
#   \code{method}
# \item \code{\link{make_preprocess}} for configuring \code{preprocess}
# \item \code{\link{input_initializers}} for configuring \code{init_inp}
# \item \code{\link{output_initializers}} for configuring \code{init_out}
# \item \code{\link{optimization_methods}} for configuring \code{opt}
# \item \code{\link{make_tricks}} for configuring \code{tricks}
# \item \code{\link{make_reporter}} for configuring \code{reporter}
# }
#
# @examples
# \dontrun{
# # Do t-SNE on the iris dataset with the same options as the t-SNE paper
# # except initialize from PCA so output is repeatable.
# # plot 2D result during embedding with convenience function for iris plot.
# # Default method is tsne. Set perplexity to 25.
# tsne_iris <- embed_prob(iris[, 1:4], opt = tsne_opt(),
#                init_inp = inp_from_perp(perplexity = 25),
#                tricks = tsne_tricks(),
#                reporter = make_reporter(plot =
#                                           make_plot(iris,
#                                                     labels = iris$Species)))
#
# # Do t-SNE on the iris dataset with the same options as the t-SNE paper
# # and initialize from a random normal distribution. Use generic plot
# # function, displaying the first two characters of the "Species" factor for
# # the points. Explicitly choose t-SNE as the method.
# tsne_iris <- embed_prob(iris[, 1:4],
#                method = tsne(),
#                opt = tsne_opt(),
#                init_inp = inp_from_perp(perplexity = 25),
#                init_out = out_from_rnorm(sd = 1e-4),
#                tricks = tsne_tricks(),
#                reporter = make_reporter(
#                  plot = make_plot(iris, label_name = "Species",
#                                   label_fn = make_label(2))))
#
# # Use the SSNE method, and preprocess input data by range scaling. t-SNE
# # tricks and optimization are reasonable defaults for other probability-based
# # embeddings. Initialize from a uniform distribution.
# ssne_iris <- embed_prob(iris[, 1:4],
#                method = ssne(),
#                opt = tsne_opt(),
#                init_inp = inp_from_perp(perplexity = 25),
#                preprocess = make_preprocess(range_scale_matrix = TRUE),
#                init_out = out_from_runif(),
#                tricks = tsne_tricks(),
#                reporter = make_reporter(
#                  plot = make_plot(iris, label_name = "Species",
#                                   label_fn = make_label(2))))
#
# # ASNE method on the s1k dataset (10 overlapping 9D Gaussian blobs),
# # Set perplexity for input initialization to 50, initialize with PCA scores,
# # preprocess by autoscaling columns, optimize
# # with Nesterov Accelrated Gradient and bold driver step size
# # (highly recommended as an optimizer). Labels for s1k are one digit, so
# # can use simplified plot function.
# asne_s1k <- embed_prob(s1k[, 1:9], method = asne(),
#  preprocess = make_preprocess(auto_scale = TRUE),
#  init_inp = inp_from_perp(perplexity = 50),
#  init_out = out_from_PCA(),
#  opt = make_opt(gradient = nesterov_gradient(), step_size = bold_driver(),
#   update = nesterov_nsc_momentum()),
#   reporter = make_reporter(plot = make_plot(s1k, label_name = "Label")))
#
# # Same as above, but using convenience method to create optimizer with less
# # typing
# asne_s1k <- embed_prob(s1k[, 1:9], method = asne(),
#  preprocess = make_preprocess(auto_scale = TRUE),
#  init_inp = inp_from_perp(perplexity = 50),
#  init_out = out_from_PCA(),
#  opt = bold_nag(),
#  reporter = make_reporter(plot = make_plot(s1k, label_name = "Label")))
# }
# @family sneer embedding functions
embed_prob <- function(xm,
                      method = tsne(),
                      preprocess = make_preprocess(verbose = verbose),
                      init_inp = inp_from_perp(perplexity = 30,
                                           input_weight_fn = exp_weight,
                                           keep_weights = method$keep_inp_weights,
                                           verbose = verbose),
                      init_out = out_from_PCA(verbose = verbose),
                      opt = mize_grad_descent(),
                      max_iter = 1000,
                      tricks = make_tricks(),
                      reporter = make_reporter(verbose = verbose),
                      export = NULL,
                      after_embed = NULL,
                      verbose = TRUE) {
  embed_main(xm, method, init_inp, init_out, opt, max_iter, tricks,
        reporter, preprocess, export, after_embed)
}

# Distance-Based Embedding.
#
# Carry out an embedding of a dataset using a distance-based method
# (e.g. Sammon Mapping), with some useful default parameters.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{distance_embedding_methods}}.
# @param preprocess Input data preprocess callback. Set by assigning the
#   result value of \code{\link{make_preprocess}}.
# @param init_inp Input initializer. Set by assigning the result value of
#   calling one of the available \code{\link{input_initializers}}.
# @param init_out Output initializer. Set by assigning the result value of
#   calling one of the available \code{\link{output_initializers}}.
# @param opt Optimization method. Set by assigning the result of value of one
#   of the configuration functions listed in
#   \code{\link{optimization_methods}}.
# @param max_iter Maximum number of optimization steps to take.
# @param tricks Optional collection of heuristics. Set by assigning the result
#  value of \code{\link{make_tricks}} or a related wrapper. See
#  \code{\link{tricks}} for the available tricks.
# @param reporter Reporter callback. Set by assigning the result value of
#   \code{\link{make_reporter}}.
# @param export Vector of names, specifying data in addition to of the output
#   data to export provide in the return value. Possible names are:
#   \describe{
#     \item{"\code{inp}"}{The input data, based on the argument passed to
#       \code{init_inp}. See the help text for the specific
#       \code{\link{input_initializers}} for details.}
#     \item{"\code{report}"}{The result of the last report. See the help
#       text for \code{\link{make_reporter}} for details.}
#     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#     passed to \code{opt} for details.}
#   }
# @param after_embed Callback to run on input and output data before output
#   data is returned.
# @param verbose If \code{TRUE} display messages about the embedding progress.
# @return The output data. A list containing:
#  \itemize{
#   \item \code{ym} Embedded coordinates.
#   \item \code{cost} The cost value associated with \code{ym}.
#   If the parameter \code{export} was used, additional elements will be
#   present. See the help text for the \code{export} parameter for more
#   details.
#  }
# @seealso
# \itemize{
# \item \code{\link{distance_embedding_methods}} for configuring
#   \code{method}
# \item \code{\link{make_preprocess}} for configuring \code{preprocess}
# \item \code{\link{input_initializers}} for configuring \code{init_inp}
# \item \code{\link{output_initializers}} for configuring \code{init_out}
# \item \code{\link{optimization_methods}} for configuring \code{opt}
# \item \code{\link{make_tricks}} for configuring \code{tricks}
# \item \code{\link{make_reporter}} for configuring \code{reporter}
# }
#
# @examples
# \dontrun{
# # Do metric MDS on the iris data set
# # In addition to the STRESS loss function also report the Kruskal Stress
# # (often used in MDS applications) and the mean relative error, which can
# # be multiplied by 100 and interpreted as a percentage error.
# mds_iris <-
#   embed_dist(iris[, 1:4],
#              method = mmds(),
#              opt = bold_nag(),
#              reporter = make_reporter(
#                          extra_costs = c("kruskal_stress",
#                                          "mean_relative_error")),
#                          plot = make_plot(iris, labels = iris$Species))
#
# # Sammon map the autoscaled iris data set, which turns out to be a
# # surprisingly tough assignment. Increase epsilon substantially to 1e-4 to
# # avoid the gradient being overwhelmed by zero distances in the input space.
# # Additionally, we report two other normalized stress functions often used
# # in MDS. The Sammon mapping cost function is already normalized, so tell the
# # make_reporter function not to report an automatically normalized version in
# # the output.
# sammon_iris <-
#   embed_dist(iris[, 1:4],
#              method = sammon_map(eps = 1e-4),
#              opt = bold_nag(),
#              preprocess = make_preprocess(auto_scale = TRUE),
#              init_out = out_from_rnorm(sd = 1e-4),
#              reporter = make_reporter(
#                 normalize_cost = FALSE,
#                 extra_costs = c("normalized_stress", "kruskal_stress"),
#                 plot = make_plot(iris, label_name = "Species",
#                                  label_fn = make_label())))
# }
# @family sneer embedding functions
embed_dist <- function(xm,
                       method = mmds(),
                       preprocess = make_preprocess(verbose = verbose),
                       init_inp = NULL,
                       init_out = out_from_PCA(verbose = verbose),
                       opt = mize_grad_descent(),
                       max_iter = 1000,
                       tricks = make_tricks(),
                       reporter = make_reporter(verbose = verbose),
                       export = NULL,
                       after_embed = NULL,
                       verbose = TRUE) {
  embed_main(xm, method, init_inp, init_out, opt, max_iter, tricks,
        reporter, preprocess, export, after_embed)
}


# Generic Embedding
#
# Carry out an embedding of a dataset using any embedding method. The most
# generic embedding function, and conversely the one with the fewest helpful
# defaults.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{embedding_methods}}.
# @param preprocess Input data preprocess callback. Set by assigning the
#   result value of \code{\link{make_preprocess}}.
# @param init_inp Input initializer. Set by assigning the result value of
#   calling one of the available \code{\link{input_initializers}}.
# @param init_out Output initializer. Set by assigning the result value of
#   calling one of the available \code{\link{output_initializers}}.
# @param opt Optimization method. Set by assigning the result of value of one
#   of the configuration functions listed in
#   \code{\link{optimization_methods}}.
# @param max_iter Maximum number of optimization steps to take.
# @param tricks Optional collection of heuristics. Set by assigning the result
#  value of \code{\link{make_tricks}} or a related wrapper. See
#  \code{\link{tricks}} for the available tricks.
# @param reporter Reporter callback. Set by assigning the result value of
#   \code{\link{make_reporter}}.
# @param export Vector of names, specifying data in addition to of the output
#   data to export provide in the return value. Possible names are:
#   \describe{
#     \item{"\code{inp}"}{The input data, based on the argument passed to
#       \code{init_inp}. See the help text for the specific
#       \code{\link{input_initializers}} for details.}
#     \item{"\code{report}"}{The result of the last report. See the help
#       text for \code{\link{make_reporter}} for details.}
#     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#     passed to \code{opt} for details.}
#   }
# @param after_embed Callback to run on input and output data before output
#   data is returned.
# @return The output data. A list containing:
#  \itemize{
#   \item \code{ym} Embedded coordinates.
#   \item \code{cost} The cost value associated with \code{ym}.
#   If the parameter \code{export} was used, additional elements will be
#   present. See the help text for the \code{export} parameter for more
#   details.
# }
# @seealso
# \itemize{
# \item \code{\link{embedding_methods}} for configuring \code{method}
# \item \code{\link{make_preprocess}} for configuring \code{preprocess}
# \item \code{\link{input_initializers}} for configuring \code{init_inp}
# \item \code{\link{output_initializers}} for configuring \code{init_out}
# \item \code{\link{optimization_methods}} for configuring \code{opt}
# \item \code{\link{make_tricks}} for configuring \code{tricks}
# \item \code{\link{make_reporter}} for configuring \code{reporter}
# }
embed_main <- function(xm, method, init_inp, init_out, opt, max_iter = 1000,
                  tricks = NULL, reporter = NULL,
                  preprocess = make_preprocess(),
                  export = NULL, after_embed = NULL) {

  opt$max_iter <- max_iter
  init_result <- init_embed(xm, method, preprocess, init_inp, init_out, opt)
  inp <- init_result$inp
  out <- init_result$out
  method <- init_result$method
  opt <- init_result$opt
  report <- init_result$report

  iter <- 0
  while (iter <= max_iter) {
    # For convenience we already did the 0 iter initialization in init_embed,
    # so we skip calling init_inp on the first iteration. Cheesy, I know.
    if (iter != 0 && !is.null(init_inp)) {
      inp_result <- init_inp(inp, method, opt, iter, out)
      inp <- inp_result$inp
      out <- inp_result$out
      method <- inp_result$method
      opt <- inp_result$opt
    }

    if (!is.null(tricks)) {
      tricks_result <- tricks(inp, out, method, opt, iter)
      inp <- tricks_result$inp
      out <- tricks_result$out
      method <- tricks_result$method
      opt <- tricks_result$opt
    }

    res <- update_out_if_necessary(inp, out, method)
    inp <- res$inp
    out <- res$out

    if (!is.null(reporter)) {
      report <- reporter(iter, inp, out, method, opt, report)
      if (report$stop_early) {
        break
      }
    }

    if (!is.null(opt$stop_early) && opt$stop_early) {
      message("Optimizer no longer making progress, stopping early")
      break
    }

    opt_result <- opt$optimize_step(opt, method, inp, out, iter)
    out <- opt_result$out
    opt <- opt_result$opt
    method <- opt_result$method
    if (!is.null(opt_result$iter)) {
      iter <- opt_result$iter
    }

    iter <- iter + 1
  }

  if (!is.null(after_embed)) {
    out <- after_embed(inp, out)
  }

  # Force an update on the report if it wasn't triggered on the final iteration
  if ((is.null(report$iter) || report$iter != iter - 1) &&
      (!report$stop_early || (!is.null(opt$stop_early) && opt$stop_early))) {
    if (!is.null(reporter)) {
      report <- reporter(iter, inp, out, method, opt, report, force = TRUE)
    }
  }

  if (!is.null(report$cost)) {
    out$cost <- report$cost
  }
  else {
    out$cost <- calculate_cost(method, inp, out)
  }

  for (obj_name in export) {
    out[[obj_name]] <- get(obj_name)
  }

  out
}

# Pre Initialization
# Runs before any input or output initialization, but from inside init_embed, so
# should always be called even if no iterations are run. Effectively also a Post
# Creation hook. Useful if there are functions that could be called during
# creation but depend on constituents that could get switched out after the
# "constructor" function is called (e.g. dynamizing a kernel)
before_init <- function(method) {
  if (!is.null(method$dynamic_kernel) && method$dynamic_kernel) {
    method <- make_kernel_dynamic(method)
  }
  if (!is.null(method$before_init_fn)) {
    method <- method$before_init_fn(method)
  }
  method
}

# Post Initialization
#
# Function called after input and output data have been initialized. Useful for
# doing data-dependent initialization of stiffness parameters, e.g. based on
# the parameterization of the input probabilities.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return List consisting of:
#  \itemize{
#   \item \code{inp} Updated input data.
#   \item \code{out} Updated output data.
#   \item \code{method} Updated embedding method.
#  }
after_init <- function(inp, out, method) {

  # Dynamic methods can affect the kernel: i.e. inhomogeneous methods enforce it
  # to be asymmetric even if all parameters are uniform after input
  # initialization, so this needs to run before we look for simpler stiffness
  # expressions
  if (!is.null(method$dyn) && !is.null(method$dyn$after_init_fn)) {
    result <- method$dyn$after_init_fn(inp, out, method)
    if (!is.null(result$inp)) {
      inp <- result$inp
    }
    if (!is.null(result$out)) {
      out <- result$out
    }
    if (!is.null(result$method)) {
      method <- result$method
    }
  }

  method <- optimize_stiffness(method)

  if (!is.null(method$after_init_fn)) {
    result <- method$after_init_fn(inp, out, method)
    if (!is.null(result$inp)) {
      inp <- result$inp
    }
    if (!is.null(result$out)) {
      out <- result$out
    }
    if (!is.null(result$method)) {
      method <- result$method
    }
  }
  list(inp = inp, out = out, method = method)
}

# Use a simplified stiffness function routine if one exists for a combination
# of kernel, cost function and normalization. Otherwise, use the generic plugin
# formulation. Runs after initialization, because some simplified stiffness
# routines are dependent on a "symmetric" kernel, which may not be determined
# until after precisions have been calculated for the input probabilities.
optimize_stiffness <- function(method) {
  if (is.null(method$stiffness)) {
    if (method$cost$name == "KL") {
      if (method$kernel$name == "exp") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- ssne_stiffness()
        }
        else if (!is_enforced_joint_out_prob(method)) {
          method$stiffness <- asne_stiffness()
        }
      }
      else if (method$kernel$name == "tdist") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- tsne_stiffness()
        }
        else {
          # t-dist kernel can never be asymmetric because it has no free
          # parameters, so for this cost/kernel combination we never need the
          # plugin stiffness
          method$stiffness <- tasne_stiffness()
        }
      }
      else if (method$kernel$name == "heavy") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- hssne_stiffness()
        }
      }
    }
    else if (method$cost$name == "revKL") {
      if (method$kernel$name == "exp") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- reverse_ssne_stiffness()
        }
        else if (!is_enforced_joint_out_prob(method)) {
          method$stiffness <- reverse_asne_stiffness()
        }
      }
      else if (method$kernel$name == "tdist") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- reverse_tsne_stiffness()
        }
      }
      else if (method$kernel$name == "heavy") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- reverse_hssne_stiffness()
        }
      }
    }
    else if (method$cost$name == "NeRV") {
      if (method$kernel$name == "exp") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- snerv_stiffness()
        }
        else if (!is_enforced_joint_out_prob(method)) {
          method$stiffness <- nerv_stiffness()
        }
      }
      else if (method$kernel$name == "tdist") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- tnerv_stiffness()
        }
      }
      else if (method$kernel$name == "heavy") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- hsnerv_stiffness()
        }
      }
    }
    else if (method$cost$name == "JS") {
      if (method$kernel$name == "exp") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- sjse_stiffness()
        }
        else if (!is_enforced_joint_out_prob(method)) {
          method$stiffness <- jse_stiffness()
        }
      }
      else if (method$kernel$name == "heavy") {
        if (is_fully_symmetric_embedding(method)) {
          method$stiffness <- hsjse_stiffness()
        }
      }
    }

    # If we get here, we've fallen through the various if-blocks: use plugin
    if (is.null(method$stiffness)) {
      method$stiffness <- plugin_stiffness()
    }
    if (method$verbose) {
      message("Using '", method$stiffness$name, "' stiffness")
    }
  }

  # Unify all 'keep' data
  keep <- method$out_keep
  for (name in c('kernel', 'cost', 'stiffness')) {
    if (!is.null(method[[name]]$keep)) {
      keep <- unique(c(keep, method[[name]]$keep))
    }
  }
  method$out_keep <- keep

  method
}

# Embedding Initialziation
#
# Initializes all data, ready to be optimized.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{embedding_methods}}.
# @param preprocess Input data preprocess callback. Does any necessary
#  filtering and scaling of the input data.
# @param init_inp Input initializer. Creates the input data required by
#  embedding algorithm.
# @param init_out Output initializer. Creates the initial set of output
#  coordinates.
# @param opt Optimization method.
# @return A list containing:
#  \itemize{
#   \item \code{inp} Initialized input.
#   \item \code{out} Initialized output.
#   \item \code{method} Initialized embedding method.
#   \item \code{opt} Initialized optimizer.
#   \item \code{report} Initialized report.
#  }
init_embed <- function(xm, method, preprocess, init_inp, init_out, opt) {

  method <- before_init(method)

  inp <- preprocess(xm)

  # Output initialization normally only needs to make us of input coordinates
  # or distance matrix, which preprocessing gives us, not things like the input
  # probabilities. So we can do output initialization here.
  out <- init_out(inp)

  # Input initialization handles input probability and other related data.
  # As we often only indirectly control the shape of the input data, we may want
  # to adjust some method parameters in the light of this data and the output
  # data. So it's most convenient to allow input initialization to assume output
  # initialization has already happened, rather than vice versa.
  if (!is.null(init_inp)) {
    inp_result <- init_inp(inp, method, opt, iter = 0, out)
    inp <- inp_result$inp
    method <- inp_result$method
    opt <- inp_result$opt
  }

  # do late initialization that relies on input or output initialization
  # being completed
  after_init_result <- after_init(inp, out, method)
  inp <- after_init_result$inp
  out <- after_init_result$out
  method <- after_init_result$method

  # initialize matrices needed for gradient calculation
  res <- update_out(inp, out, method)
  inp <- res$inp
  out <- res$out

  # reuse reports from old invocation of reporter, so we can use info
  # to determine whether to stop early (e.g. relative convergence tolerance)
  report <- list( stop_early = FALSE )

  list(
    inp = inp,
    out = out,
    method = method,
    opt = opt,
    report = report
  )
}

# Set Output Data Coordinates
#
# This function sets the embedded coordinates in the output data, as well as
# recalculating any auxiliary output data that is dependent on the coordinates
# (e.g. distances and probabilities)
#
# @param inp Input data.
# @param coords Matrix of coordinates.
# @param method Embedding method.
# @param mat_name Name of the \code{out} entry to store \code{coords} in.
#  Defaults to \code{ym}.
# @param out Existing output data. Optional.
# @return \code{out} list with updated with coords and auxiliary data.
set_solution <- function(inp, coords, method, mat_name = "ym",
                         out = NULL) {
  if (is.null(out)) {
    out <- list()
  }
  out[[mat_name]] <- coords
  res <- update_out(inp, out, method)
  inp <- res$inp
  out <- res$out
  list(out = out, inp = inp)
}

# Check if Output Data Needs Updating
#
# When the coordinates of an embedding are updated via
# \code{\link{set_solution}}, the other matrices and values that depend on
# those coordinates should be updated automatically. If a change has occurred
# to another part of the embedding data (most likely the input probabilities,
# or the output kernel parameters), then you may need to call this function to
# see if the output data needs updating before calculating the cost or
# gradient.
#
# @param out Output data.
# @return \code{TRUE} if the output data matrices need updating.
should_update <- function(out) {
  !is.null(out$dirty) && out$dirty
}

# Mark Output Data as Up to Date
#
# This function should be called when the output data has just been updated.
# Some parts of the code will avoid unnecessarily repeating the update, but
# only if they can detect that there's no change to make.
#
# @param out Output data.
# @return Output data, now marked as being up to date.
undirty <- function(out) {
  out$dirty <- FALSE
  out
}

# Update if Something Has Changed
#
# This function calls \code{\link{update_out}} only if it detects that there
# are changes that needs to be made.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return (Possible) updated output data. Otherwise, \code{out} is returned
# unchanged.
update_out_if_necessary <- function(inp, out, method) {
  if (should_update(out)) {
    res <- update_out(inp, out, method)
    inp <- res$inp
    out <- res$out
  }
  list(inp = inp, out = out)
}

# Update Output Data
#
# Updates the output data (matrices dependent on the embedding coordinates,
# such as the distance matrix, weight matrix, output probability matrix).
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Updated output data.
update_out <- function(inp, out, method) {
  if (!is.null(method$update_out_fn)) {
    res <- method$update_out_fn(inp, out, method)
    if (!is.null(res$inp)) {
      inp <- res$inp
    }
    if (!is.null(res$out)) {
      out <- res$out
    }
    out <- undirty(out)
  }
  list(inp = inp, out = out)
}

