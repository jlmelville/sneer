# Functions to be run at a fixed frequency during the optimization.

#' Reporter
#'
#' Factory function which returns a function which will be invoked by the
#' embedding algorithm at regular intervals during the optimization.
#'
#' @param report_every Number of steps between callback invocations.
#' @param min_cost If the cost function used by the embedding falls below this
#'  value, the optimization process is halted.
#' @param reltol If the relative tolerance of the cost function between
#'  consecutive reports falls below this value, the optimization process is
#'  halted.
#' @param disttol If the RMSD between output distance matrices calculated
#'  between consecutive reports falls below this value, the optimization
#'  process is halted.
#' @param plot Function for plotting embedding. Signature should be
#'  \code{plot(out)} where \code{out} is the output data list. Return value
#'  of this function is ignored.
#' @param normalize_cost If \code{TRUE}, the cost calculated by the cost
#'  function will be normalized and both values logged.
#' @param keep_costs If \code{TRUE}, all costs (including those specified in
#'  \code{extra_costs}) and the iteration number at
#'  which they were calculated will be stored on result list returned by the
#'  reporter callback.
#' @param extra_costs List containing the names of cost functions to be
#'  reported, in addition to the cost associated with the embedding method,
#'  which will always be logged. Possible cost functions include:
#'  \describe{
#'    \item{\code{"kl"}}{\code{\link{kl_cost}}.}
#'    \item{\code{"kruskal_stress"}}{\code{\link{kruskal_stress_cost}}.}
#'    \item{\code{"mean_relative_error"}}{\code{\link{mean_relative_error_cost}}.}
#'    \item{\code{"metric_sstress"}}{\code{\link{metric_sstress_cost}}.}
#'    \item{\code{"metric_stress"}}{\code{\link{metric_stress_cost}}.}
#'    \item{\code{"normalized_stress"}}{\code{\link{normalized_stress_cost}}.}
#'    \item{\code{"rms_metric_stress"}}{\code{\link{rms_metric_stress_cost}}.}
#'    \item{\code{"sammon_stress"}}{\code{\link{sammon_stress_cost}}.}
#'  }
#'  Note that not all costs are compatible with all embedding methods, because
#'  they may require specific matrices or other values to be precalculated in
#'  the Input and Output data. See the help text for each cost function for
#'  details.
#' @param opt_report If \code{TRUE}, summary of the state of the optimization
#'  (e.g. step size, momentum, gradient length) will be reported.
#' @param out_report If \code{TRUE}, summary of the state of the output data
#'  (e.g. coordinates, probabilities, weights) will be reported.
#' @param verbose If \code{TRUE}, report results such as cost values
#'  will be logged to screen. If set to false, you will have to export the
#'  report from the embedding routine to access any of the information.
#' @return Reporter callback used by the embedding routine. The callback has the
#' signature \code{reporter(iter, inp, out, method, report)} where:
#'  \item{\code{iter}}{Iteration number.}
#'  \item{\code{inp}}{Input data.}
#'  \item{\code{method}}{Embedding method.}
#'  \item{\code{report}}{Report from previous reporter invocation.}
#'  \item{\code{force}}{If TRUE, then the report will be generated even if the
#'  iteration number doesn't meet the \code{report_every} criterion.}
#' The return value of the callback is an updated
#' version of the \code{report} passed as a parameter to the callback. A list
#' containing:
#'  \item{\code{stop_early}}{If \code{TRUE} then optimization stopped before
#'  the maximum number of iterations specified by the embedding algorithm.}
#'  \item{\code{reltol}}{Relative convergence value.}
#'  \item{\code{iter}}{Iteration number that the callback was invoked at.}
#'  \item{\code{cost}}{Cost for this reporter.}
#'  \item{\code{norm}}{Normalized cost for the most recent iteration. Only
#'  present if \code{normalize_cost} was \code{TRUE}.}
#'  \item{\code{costs}}{Matrix of all costs calculated for all invocations of
#'  the reporter callback and the value of \code{iter} the reporters were
#'  invoked at. Only present if \code{keep_costs} was \code{TRUE}.}
#'
#' The result list is reused on each invocation of the callback so that the cost
#' from the previous report can be compared with that of the current report,
#' allowing for relative convergence early stopping, and appending of costs
#' if \code{keep_costs} is \code{TRUE}.
#' @seealso \code{\link{embed_prob}} for how to use this function for configuring
#' an embedding, and \code{\link{make_plot}} for 2D plot generation.
#' @examples
#' # reporter calculation every 100 steps of optimization, log cost and also the
#' # normalized cost
#' make_reporter(report_every = 100, normalize_cost = TRUE)
#'
#' # Stop optimization early if relative tolerance of costs falls below 0.001
#' make_reporter(report_every = 100, reltol = 0.001)
#'
#' # For s1k dataset, plot 2D embedding at every reporter, with "Label" factor
#' # to identify each point on the plot
#' make_reporter(report_every = 100, plot = make_plot(s1k, "Label"))
#'
#' # For iris dataset, plot 2D embedding at every reporter, with first two
#' # characters of the "Species" factor to identify each point on the plot
#' make_reporter(report_every = 100,
#'               plot = make_plot(iris, "Species", make_label(2)))
#'
#' # Keep all costs calculated during reporters, can be exported from the
#' # embedding routine and plotted or otherwise used.
#' make_reporter(report_every = 100, keep_costs = TRUE)
#'
#' # Report normalized stress, Kruskal stress and Sammon stress:
#' make_reporter(extra_costs =
#'                 c("normalized_stress", "kruskal_stress", "sammon_stress"))
#'
#' # Should be passed to the reporter argument of an embedding function:
#' \dontrun{
#'  embed_prob(reporter = make_reporter(report_every = 100,
#'                                     normalize_cost = TRUE,
#'                                     plot = make_plot(iris, "Species")),
#'                                     ...)
#' }
#' @export
make_reporter <- function(report_every = 100, min_cost = 0,
                          reltol = sqrt(.Machine$double.eps),
                          disttol = sqrt(.Machine$double.eps),
                          plot = NULL,
                          normalize_cost = TRUE, keep_costs = FALSE,
                          extra_costs = NULL, opt_report = FALSE,
                          out_report = FALSE,
                          verbose = TRUE) {
  reporter <- list()

  reporter$cost_log <- function(iter, inp, out, method, opt, result) {
    cost <- calculate_cost(method, inp, out)
    if (normalize_cost) {
      norm_fn <- make_normalized_cost_fn(method$cost$fn)
      norm_cost <- norm_fn(inp, out, method)
    }

    if (verbose) {
      cost_str <- paste0(" cost = ", formatC(cost))
      if (normalize_cost) {
        cost_str <- paste0(cost_str, " norm = ", formatC(norm_cost))
      }
    }

    if (!is.null(extra_costs)) {
      for (extra_cost_name in extra_costs) {
        # append "_cost" to the name in the array to get the actual func name
        extra_cost_fn <- get(paste0(extra_cost_name, "_cost"))
        extra_cost <- extra_cost_fn(inp, out, method)
        if (verbose) {
          cost_str <- paste0(cost_str, " ", extra_cost_name, " = ",
                             formatC(extra_cost))
        }
        result[[extra_cost_name]] <- extra_cost
      }
    }

    rtol <- NULL
    if (!is.null(result$cost)) {
      rtol <- reltol(cost, result$cost)
      if (verbose) {
        cost_str <- paste0(cost_str, " rtol = ", formatC(rtol))
      }
      result$reltol <- rtol
    }

    rmsd <- NULL
    if (is.null(result$dm)) {
      if (!is.null(out$dm)) {
        result$dm <- upper_tri(out$dm)
      }
      else {
        result$dm <- upper_tri(distance_matrix(out$ym))
      }
    }
    else {
      if (!is.null(out$dm)) {
        dm <- upper_tri(out$dm)
      }
      else {
        dm <- upper_tri(distance_matrix(out$ym))
      }
      rmsd <- sqrt(sum((result$dm - dm) ^ 2) / length(dm))
      result$dm <- dm
      if (verbose) {
        cost_str <- paste0(cost_str, " dtol = ", formatC(rmsd))
      }
    }

    if (verbose) {
      message("Iteration #", iter, cost_str)
      flush.console()
    }

    if (cost < min_cost) {
      if (verbose) {
        message("minimum cost ", formatC(min_cost), " convergence reached")
      }
      result$stop_early <- TRUE
    }
    else if (!is.null(rtol) && rtol < reltol) {
      if (verbose) {
        message("relative tolerance ", formatC(reltol), " convergence reached")
      }
      result$stop_early <- TRUE
    }
    else if (!is.null(rmsd) && rmsd < disttol) {
      if (verbose) {
        message("distance tolerance ", formatC(rmsd), " convergence reached")
      }
      result$stop_early <- TRUE
    }

    result$cost <- cost

    if (normalize_cost) {
      result$norm <- norm_cost
    }

    if (keep_costs) {
      if (normalize_cost) {
        names <- c("iter", "cost", "norm", extra_costs)
      } else {
        names <- c("iter", "cost", extra_costs)
      }
      cost_row <- matrix(unlist(result[names]), nrow = 1)
      result$costs <- rbind(result$costs, cost_row)
      colnames(result$costs) <- names
    }

    result
  }

  if (!is.null(plot)) {
    reporter$plot_embedding <- function(iter, inp, out, method, opt, result) {
      plot(out)
      result
    }
  }

  if (opt_report) {
    reporter$opt_report <- function(iter, inp, out, method, opt, result) {
      opt_data <- opt$report(opt)
      opt_str <- ""
      for (name in names(opt_data)) {
        opt_str <- paste0(opt_str, name, ": ", formatC(opt_data[[name]]), " ")
      }
      if (nchar(opt_str) > 0) {
        message(opt_str)
      }
      result
    }
  }

  if (out_report) {
    reporter$out_report <- function(iter, inp, out, method, opt, result) {
      for (name in names(out)) {
        if (class(out[[name]]) == "matrix") {
          summarize(out[[name]], name)
        }
      }
      result
    }
  }

  function(iter, inp, out, method, opt, report, force = FALSE) {
    report$stop_early <- FALSE
    if (iter %% report_every == 0 || force) {
      report$iter <- iter
      for (name in names(reporter)) {
        report <- reporter[[name]](iter, inp, out, method, opt, report)
      }
    }
    report
  }
}
