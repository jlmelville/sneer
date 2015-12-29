# Miscellaneous data manipulations that may help optimize the embedding.
# If you are fancy, you could call them "heuristics", but the t-SNE paper calls
# their approach "tricks" and that's less characters to type.

#' Tricks
#'
#' Creates a callback which the embedding routine will call before each
#' optimization step.
#'
#' @param early_exaggeration If \code{TRUE}, then apply early exaggeration.
#' @param P_exaggeration The size of the exaggeration.
#' @param exaggeration_off_iter Iteration step to turn exaggeration off.
#' @param verbose If \code{TRUE} log messages about trick application.
#' @return Tricks callback with the signature \code{tricks(inp, out, method,
#' opt, iter)} where:
#' \describe{
#'  \item{\code{inp}}{Input data.}
#'  \item{\code{out}}{Output data.}
#'  \item{\code{method}}{Embedding method.}
#'  \item{\code{opt}}{Optimizer.}
#'  \item{\code{iter}}{Iteration number.}
#' }
#' and returns a list containing:
#' \describe{
#'  \item{\code{inp}}{Updated input data.}
#'  \item{\code{out}}{Updated output data.}
#'  \item{\code{method}}{Updated embedded method.}
#'  \item{\code{opt}}{Updated optimizer.}
#' }
#' @seealso \code{\link{embed_prob}} for how to use this function for
#' configuring an embedding.
#' @examples
#' # Use early exaggeration as described in the t-SNE paper
#' make_tricks(early_exaggeration = TRUE, P_exaggeration = 4,
#'             exaggeration_off_iter = 50)
#'
#' # Should be passed to the tricks argument of an embedding function:
#' \dontrun{
#'  embed_prob(tricks = make_tricks(early_exaggeration = TRUE,
#'                                 P_exaggeration = 4,
#'                                 exaggeration_off_iter = 50), ...)
#' }
#' @family sneer tricks
#' @export
make_tricks <- function(early_exaggeration = TRUE, P_exaggeration = 4,
                        exaggeration_off_iter = 50, verbose = TRUE) {
  tricks <- list()
  if (early_exaggeration) {
    exaggeration_func <- function(inp, out, method, opt, iter) {
      if (iter == 0) {
        inp$pm <- inp$pm * P_exaggeration
      }
      if (iter == exaggeration_off_iter) {
        if (verbose) {
          message("Exaggeration off at iter: ", iter)
        }
        inp$pm <- inp$pm / P_exaggeration
      }

      list(inp = inp)
    }
    tricks$exaggeration <- exaggeration_func
  }

  function(inp, out, method, opt, iter) {
    for (name in names(tricks)) {
      result <- tricks[[name]](inp, out, method, opt, iter)
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
#' Laurens van der Maarten, Geoffrey Hinton.
#' Visualizing Data using t-SNE.
#' Journal of Machine Learning Research, 2008, 9, 2579-2605.
#' @family sneer tricks
#' @export
tsne_tricks <- function(verbose = TRUE) {
  make_tricks(early_exaggeration = TRUE, P_exaggeration = 4,
              exaggeration_off_iter = 50, verbose = verbose)
}
