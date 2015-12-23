# Functions to be run at a fixed frequency during the optimization.

#' Create a reporter callback.
#'
#' Factory function which returns a function which will be invoked by the
#' embedding algorithm at regular intervals during the optimization.
#'
#' The result of this function is a callback with the signature
#' \code{reporter(iter, inp, out, method, report)} where:
#' \itemize{
#'  \item \code{iter} Iteration number.
#'  \item \code{inp} Input data.
#'  \item \code{method} Embedding method.
#'  \item \code{report} Report from previous reporter invocation.
#' }
#'
#' The return value of the callback is an updated
#' version of the \code{report} passed as a parameter to the callback. It has
#' the following elements:
#' \itemize{
#'  \item \code{stop_early} If \code{TRUE} then optimization stopped before
#'  the maximum number of iterations specified by the embedding algorithm.
#'  \item \code{reltol} Relative convergence value.
#'  \item \code{iter} Iteration number that the callback was invoked at.
#'  \item \code{cost} Cost for this reporter.
#'  \item \code{norm} Normalized cost for the most recent iteration. Only
#'  present if \code{normalize_cost} was \code{TRUE}.
#'  \item \code{costs} Matrix of all costs calculated for all invocations of
#'  the reporter callback and the value of \code{iter} the reporters were
#'  invoked at.
#'  Only present if \code{keep_costs} was \code{TRUE}.
#' }
#'
#' The result list is reused on each invocation of the callback so that the cost
#' from the previous report can be compared with that of the current report,
#' allowing for relative convergence early stopping, and appending of costs
#' if \code{keep_costs} is \code{TRUE}.
#'
#' @param report_every Number of steps between callback invocations.
#' @param min_cost If the cost function used by the embedding falls below this
#' value, the optimization process is halted.
#' @param reltol If the relative tolerance of the cost function between
#' consecutive reports falls below this value, the optimization process is
#' halted.
#' @param plot_fn Function for plotting embedding. Signature should be
#' \code{plot_fn(out)} where \code{out} is the output data list. Return value
#' of this function is ignored.
#' @param normalize_cost If \code{TRUE}, the cost calculated by the cost
#' function will be normalized and both values logged.
#' @param keep_costs If \code{TRUE}, all costs (and the iteration number at
#' which they were calculated) will be stored on result list returned by the
#' reporter callback. This can be exported by the embedding algorithm.
#' @param verbose If \code{TRUE}, report results such as cost values
#' will be logged to screen. If set to false, you will have to export the
#' report from the embedding routine to access any of the information.
#' @return reporter callback used by the embedding routine.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
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
#' make_reporter(report_every = 100, plot_fn = make_plot(s1k, "Label"))
#'
#' # For iris dataset, plot 2D embedding at every reporter, with first two
#' # characters of the "Species" factor to identify each point on the plot
#' make_reporter(report_every = 100,
#'               plot_fn = make_plot(iris, "Species", make_label(2)))
#'
#' # Keep all costs calculated during reporters, can be exported from the
#' # embedding routine and plotted or otherwise used.
#' make_reporter(report_every = 100, keep_costs = TRUE)
#'
#' # Should be passed to the reporter argument of an embedding function:
#' \dontrun{
#'  embed_sim(reporter = make_reporter(report_every = 100,
#'                                     normalize_cost = TRUE,
#'                                     plot_fn = make_plot(iris, "Species")),
#'                                     ...)
#' }
#' @export
make_reporter <- function(report_every = 100, min_cost = 0,
                          reltol = sqrt(.Machine$double.eps), plot_fn = NULL,
                          normalize_cost = TRUE, keep_costs = FALSE,
                          verbose = TRUE) {
  reporter <- list()

  reporter$cost_log <- function(iter, inp, out, method, result) {
    cost <- method$cost_fn(inp, out)
    if (is.null(result$cost)) {
      result$cost <- .Machine$double.xmax
    }

    if (normalize_cost) {
      norm_fn <- make_normalized_cost_fn(method$cost_fn)
      norm_cost <- norm_fn(inp, out)
    }

    if (verbose) {
      cost_str <- paste0(" cost = ", formatC(cost))
      if (normalize_cost) {
        cost_str <- paste0(cost_str, " norm = ", formatC(norm_cost))
      }
      message("Iteration #", iter, cost_str)
      flush.console()
    }
    if (cost < min_cost) {
      if (verbose) {
        message("minimum cost ", formatC(min_cost), " convergence reached")
      }
      result$stop_early <- TRUE
    } else {
      rtol <- reltol(cost, result$cost)
      if (rtol < reltol) {
        if (verbose) {
          message("relative tolerance ", formatC(reltol),
                  " convergence reached")
        }
        result$stop_early <- TRUE
      }
      result$reltol <- rtol
    }

    result$cost <- cost
    if (keep_costs) {
      result$costs <- rbind(result$costs, matrix(c(iter, cost), nrow = 1))
      colnames(result$costs) <- c("iter", "cost")
    }

    if (normalize_cost) {
      result$norm <- norm_cost
    }

    result
  }

  if (!is.null(plot_fn)) {
    reporter$plot_embedding <- function(iter, inp, out, method, result) {
      plot_fn(out)
      result
    }
  }

  function(iter, inp, out, method, report) {
    report$stop_early <- FALSE
    if (iter %% report_every == 0) {
      for (name in names(reporter)) {
        report <- reporter[[name]](iter, inp, out, method, report)
      }
      report$iter <- iter
    }
    report
  }
}
