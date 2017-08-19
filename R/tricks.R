# Tricks
#
# Miscellaneous data manipulations that may help optimize the embedding.
# If you are fancy, you could call them "heuristics", but the t-SNE paper
# calls their approach "tricks" and that's less characters to type, so we'll
# go with that too.
#
# @examples
# # pass one or more tricks to the make_tricks factory function:
# # no tricks
# make_tricks()
#
# # one trick
# make_tricks(early_exaggeration())
#
# # two tricks
# make_tricks(early_exaggeration(), late_momentum())
#
# \dontrun{
# # in turn, pass the result of make_tricks to an embedding routine
# embed_prob(make_tricks(early_exaggeration(), late_momentum()), ...)
# }
#
# @keywords internal
# @name tricks
# @family sneer tricks
NULL


# Tricks
#
# A collection of heuristics to improve embeddings.
#
# Creates a callback which the embedding routine will call before each
# optimization step. Represents a collection of miscellaneous tweaks that
# don't fit neatly into the rest of the framework, and as such can modify
# the state of any other component of the embedding routine: input data,
# output data, optimizer or embedding method.
#
# The input to this function can be zero, one or multiple independent tricks,
# each of which is created by their own factory function. As a result, the
# signature of this function is not informative. For information on the tricks
# available, see \code{tricks}.
#
# Some wrappers around this function which apply sets of tricks from the
# literature are available. See the links under the 'See Also' section.
#
# @param ... Zero or more \code{tricks}.
# @return Callback collecting all the supplied tricks, to be invoked by the
# embedding routine.
# @seealso The result value of this function should be passed to the
# \code{tricks} parameter of embedding routines like
# \code{embed_prob} and \code{embed_dist}.
# @examples
# # Use early exaggeration as described in the t-SNE paper
# make_tricks(early_exaggeration(exaggeration = 4, off_iter = 50))
#
# # Should be passed to the tricks argument of an embedding function:
# \dontrun{
#  embed_prob(tricks = make_tricks(early_exaggeration(
#                                  exaggeration = 4, off_iter = 50), ...)
# }
# @family sneer trick collections
make_tricks <- function(...) {
  tricks <- list(...)

  function(inp, out, method, opt, iter) {
    inp$dirty <- FALSE
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

# Early Exaggeration
#
# A "trick" for improving the quality of the embedding.
#
# This trick is for use with probability-based embedding. It scales up the
# input probabilities for the first few iterations of the embedding to
# encourage close distances to form between very similar points.
#
# @param exaggeration Size of the exaggeration factor: input probabilities will
# be multiplied by this value.
# @param off_iter Iteration step at which the input probabilities are returned
# to their original values.
# @param verbose If \code{TRUE} report a message when exaggeration is turned
# off.
# @return Trick callback. Should be passed to \code{make_tricks} when
# configuring an embedding.
# @examples
# \dontrun{
# # exaggerate for the first 100 iterations
# embed_prob(make_tricks(early_exaggeration(off_iter = 100)), ...)
# }
# @family sneer tricks
early_exaggeration <- function(exaggeration = 4, off_iter = 50,
                               verbose = TRUE) {
  function(inp, out, method, opt, iter) {
    dirty <- FALSE
    if (iter == 0) {
      inp$pm <- inp$pm * exaggeration
      dirty <- TRUE
    }
    if (iter == off_iter) {
      if (verbose) {
        message("Exaggeration off at iter: ", iter)
      }
      inp$pm <- inp$pm / exaggeration
      dirty <- TRUE
    }

    if (dirty) {
      upd_res <- inp_updated(inp, out, method, iter)
      inp <- upd_res$inp
      out <- upd_res$out
      method <- upd_res$method
    }

    list(inp = inp, out = out, method = method)
  }
}

# t-SNE Tricks
#
# Tricks configured according to the details in the t-SNE paper.
#
# @param verbose If \code{TRUE} print message about tricks during the
# embedding.
# @return tricks callback parameterized to behave like the t-SNE paper.
# @seealso \code{embed_prob} for how to use this function for
# configuring an embedding.
# @examples
# # Should be passed to the tricks argument of an embedding function:
# \dontrun{
#  embed_prob(tricks = tsne_tricks(), ...)
# }
# @references
# Van der Maaten, L., & Hinton, G. (2008).
# Visualizing data using t-SNE.
# \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
# @family sneer trick collections
tsne_tricks <- function(verbose = TRUE) {
  make_tricks(early_exaggeration(exaggeration = 4, off_iter = 50,
                                 verbose = verbose))
}
