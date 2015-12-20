# Miscellaneous data manipulations that may help optimize the embedding.
# If you are fancy, you could call them "heuristics", but the t-SNE paper calls
# their approach "tricks" and that's less characters to type.

#' Create tricks callback.
#'
#' Creates a callback which the embedding routine will call before each
#' optimization step. The callback has the signature
#' \code{inp, out, method, opt, iter} where:
#' \itemize{
#'  \item \code{inp} Input data.
#'  \item \code{out} Output data.
#'  \item \code{method} Embedding method.
#'  \item \code{opt} Optimizer.
#'  \item \code{iter} Iteration number.
#' }
#' and returns a list containing:
#' \itemize{
#'  \item \code{inp} Updated input data.
#'  \item \code{out} Updated output data.
#'  \item \code{method} Updated embedded method.
#'  \item \code{opt} Updated optimizer.
#' }
#'
#' @param early_exaggeration If \code{TRUE}, then apply early exaggeration.
#' @param P_exaggeration The size of the exaggeration.
#' @param exaggeration_off_iter Iteration step to turn exaggeration off.
#' @param verbose If \code{TRUE} log messages about trick application.
#' @return Tricks callback.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding, and \code{\link{tsne_tricks}} for a convenience function that
#' supplies the parameters used in the t-SNE paper.
#' @examples
#' # Use early exaggeration as described in the t-SNE paper
#' make_tricks(early_exaggeration = TRUE, P_exaggeration = 4,
#'             exaggeration_off_iter = 50)
#' # Should be passed to the tricks argument of an embedding function:
#' \dontrun{
#'  embed_sim(tricks = make_tricks(early_exaggeration = TRUE,
#'                                 P_exaggeration = 4,
#'                                 exaggeration_off_iter = 50), ...)
#' }
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

#' Tricks from t-SNE paper.
#'
#' @param verbose If \code{TRUE} print message about tricks during the
#' embedding.
#' @return tricks callback parameterized to behave like the t-SNE paper.
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding, and \code{\link{make_tricks}} for a more generic function.
#' @examples
#' # Should be passed to the tricks argument of an embedding function:
#' \dontrun{
#'  embed_sim(tricks = tsne_tricks(), ...)
#' }
tsne_tricks <- function(verbose = TRUE) {
  make_tricks(early_exaggeration = TRUE, P_exaggeration = 4,
              exaggeration_off_iter = 50, verbose = verbose)
}
