# Functions to be run once every "epoch", although in the context of embedding
# this just means a fixed number of steps of optimization.

#' Create an epoch callback.
#'
#' Factory function which returns a function which will be invoked by the
#' embedding algorithm at regular intervals during the optimization.
#'
#' The result of invoking the callback that this function creates is a list
#' with the following items:
#'
#' \itemize{
#'  \item \code{stop_early} If \code{TRUE} then optimization stopped before
#'  the maximum number of iterations specified by the embedding algorithm.
#'  \item \code{reltol} Relative convergence value.
#'  \item \code{iter} Iteration number that the callback was invoked at.
#'  \item \code{cost} Cost for this epoch.
#'  \item \code{stress} Stress value (scaled cost value) for this epoch. Only
#'  present if \code{calc_stress} was \code{TRUE}.
#'  \item \code{costs} Matrix of all costs calculated for all invocations of
#'  the epoch callback and the value of \code{iter} the epochs were invoked at.
#'  Only present if \code{save_costs} was \code{TRUE}.
#' }
#'
#' @param epoch_every Number of steps between callback invocations.
#' @param min_cost If the cost function used by the embedding falls below this
#' value, the optimization process is halted.
#' @param reltol If the relative tolerance of the cost function between
#' invocations of the epoch function falls below this value, the optimization
#' process if halted.
#' @param plot_func Function for plotting embedding. Signature should be
#' \code{plot_fn(out)} where \code{out} is the output data list. Return value
#' of this function is ignored.
#' @param calc_stress If \code{TRUE}, the cost calculated by the cost function
#' will be converted to a stress and both values logged.
#' @param save_costs If \code{TRUE}, all costs (and the iteration number at
#' which they were calculated) will be stored on result list returned by the
#' epoch callback. This can be exported by the embedding algorithm.
#' @param verbose If \code{TRUE}, epoch callback results such as cost values
#' will be logged to screen. If set to false, you will have to export the
#' epoch result from the embedding routine to access any of the information.
#' @return an epoch callback used by the embedding routine. It has the signature
#' \code{epoch_callback(iter, inp, out, stiffness, result)} where \code{iter} is
#' the iteration number, \code{inp} is the input data list, \code{out} is the
#' output data list, \code{stiffness} is stiffness data list and \code{result}
#' is the epoch result list. The return value of the callback is an updated
#' version of the \code{result}. The result list is reused on each invocation
#' of the callback so that the cost from the previous epoch can be compared
#' with that of the current epoch, allowing for relative convergence early
#' stopping, and appending of costs if \code{save_costs} is \code{TRUE}.
make_epoch <- function(epoch_every = 100, min_cost = 0,
                       reltol = sqrt(.Machine$double.eps), plot_func = NULL,
                       calc_stress = TRUE, save_costs = FALSE,
                       verbose = TRUE) {
  epoch <- list()

  epoch$cost_log <- function(iter, inp, out, stiffness, result) {
    cost <- stiffness$cost_fn(inp, out)
    if (is.null(result$cost)) {
      result$cost <- .Machine$double.xmax
    }

    if (calc_stress) {
      stress_fn <- make_stress_fn(stiffness$cost_fn)
      stress <- stress_fn(inp, out)
    }

    if (verbose) {
      cost_str <- paste0(" cost = ", formatC(cost))
      if (calc_stress) {
        cost_str <- paste0(cost_str, " stress = ", formatC(stress))
      }
      message("Epoch: Iteration #", iter, cost_str)
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
    if (save_costs) {
      result$costs <- rbind(result$costs, matrix(c(iter, cost), nrow = 1))
      colnames(result$costs) <- c("iter", "cost")
    }

    if (calc_stress) {
      result$stress <- stress
    }

    result
  }

  if (!is.null(plot_func)) {
    epoch$plot_embedding <- function(iter, inp, out, stiffness, result) {
      plot_func(out)
      result
    }
  }

  function(iter, inp, out, stiffness, result) {
    result$stop_early <- FALSE
    if (iter %% epoch_every == 0) {
      for (name in names(epoch)) {
        result <- epoch[[name]](iter, inp, out, stiffness, result)
      }
      result$iter <- iter
    }
    result
  }
}
