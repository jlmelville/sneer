# Functions for initializing the input data.

#' Input Initialization
#'
#' Factory function for input initialization.
#'
#' This function is responsible for creating a method to initialize the input
#' data for embedding. It generates a distance matrix if the input data was
#' in the form of coordinates. Additionally, if any
#' \code{\link{input_initializers}} are passed as parameters to this function,
#' these will be invoked in order they were passed.
#'
#' @param ... Zero or more \code{\link{input_initializers}}.
#' @return An input initializer to be used by an embedding routine. The input
#'   data is not of direct interest in most embeddings, but may be useful for
#'   diagnostic purposes. If exported from an embedding, it will be as a list
#'   called \code{inp} on the embedding result, and itself contains:
#'   \item{\code{xm}}{Input coordinates if these were provided.}
#'   \item{\code{dm}}{Input distances.}
#'   If other initializers were provided as arguments to this method, they may
#'   provide extra data - see the documentation of their functions for details.
#' @seealso \code{\link{embed_dist}} and \code{\link{embed_prob}}
#'   for how to use this function to configure an embedding and to export input
#'   data. Also see the documentation for the specific members of the
#'   \code{\link{input_initializers}} that can be passed to this function.
#'
#' @examples
#' # Should be passed to the init_inp argument of an embedding function:
#' \dontrun{
#'  # distance-based embedding don't need extra input initialization
#'  embed_dist(init_inp = make_init_inp(), method = mds())
#'
#'  # probability-based embeddings must also initialize input probabilities
#'  embed_prob(init_inp = make_init_inp(prob_perp_bisect()), method = tsne())
#' }
#' @export
make_init_inp <- function(...) {
  init_inp <- list(...)

  function(xm) {
    inp <- list()

    if (class(xm) == "dist") {
      inp$dm <- as.matrix(xm)
    } else {
      inp$xm <- as.matrix(xm)
      inp$dm <- distance_matrix(inp$xm)
    }

    for (i in seq_along(init_inp)) {
      inp <- init_inp[[i]](inp)
    }

    inp
  }
}

#' Input Initializers
#'
#' These methods deal with converting the input coordinates or distances
#' into the structures required by different embedding methods. For instance,
#' probability-based embedding methods require the construction of a probability
#' matrix from the input distances.
#'
#' @seealso Input initializers should be passed to the
#' \code{\link{make_init_inp}} function.
#'
#' @examples
#' # initializer that uses bisection search to create input probability
#' # distribution with a perplexity of 50 - pass to make_init_inp:
#' make_init_inp(prob_perp_bisect(perplexity = 50))
#'
#' \dontrun{
#' # in turn, pass the result of make_init_inp to an embedding routine
#' embed_prob(method = tsne(),
#'            init_inp = make_init_inp(prob_perp_bisect(perplexity = 50)))
#' }
#' @keywords internal
#' @name input_initializers
#' @family sneer input initializers
NULL

#' Input Probability Initialization By Bisection Search on Perplexity
#'
#' An initialization method for creating input probabilities.
#'
#' Function to generate a row probability matrix by optimizing a one-parameter
#' weighting function. This is used to create input probabilities from the
#' input distances, such that each row of the matrix is a probability
#' distribution with the specified perplexity.
#'
#' This is the method described in the original SNE paper, and few methods
#' deviate very strongly from it, although they may do further processing
#' on the resulting probability matrix. For example, SSNE and t-SNE convert
#' this matrix into a single joint probability distribution.
#' @param perplexity Target perplexity value for the probability distributions.
#' @param input_weight_fn Weighting function for distances. It should have the
#'   signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix
#'   of squared distances and \code{beta} is a real-valued scalar parameter
#'   which will be varied as part of the search to produce the desired
#'   \code{perplexity}. The function should return a matrix of weights
#'   corresponding to the transformed squared distances passed in as arguments.
#' @param keep_all_results If \code{true} then the list returned by the callback
#'   will also contain a vector of \code{beta} parameters that generated the
#'   probability matrix. Otherwise, only the probability matrix is returned.
#' @param verbose If \code{TRUE} display messages about progress of
#'   initialization.
#' @return Input initialization method for use by \code{\link{make_init_inp}}.
#'   Although direct use of this value is intended only by the embedding
#'   function, the resulting data can be accessed by passing \code{"inp"} to
#'   the \code{export} list in an embedding function such as
#'   \code{\link{embed_prob}} or \code{\link{embed_dist}}. If exported, the
#'   \code{inp} list will contain the follow data created by this method:
#'   \item{\code{pm}}{Input probabilities.}
#'   \item{\code{beta}}{Input weighting parameters that produced the
#'     probabilities. Only provided if \code{keep_all_results} is \code{TRUE}.}
#' @seealso \code{\link{make_init_inp}} for how to use this function as part
#' of an embedding.
#' @examples
#' # Set target perplexity of each probability distribution to 30
#' prob_perp_bisect(perplexity = 30)
#'
#' # Set target perplexity of each probability distribution to 30 but use
#' # a different weighting function.
#' prob_perp_bisect(perplexity = 30, input_weight_fn = sqrt_exp_weight)
#'
#' # Perplexity of 50, and keep the values of the exponential parameter for
#' # later processing or reporting.
#' prob_perp_bisect(perplexity = 50, keep_all_results = TRUE)
#'
#' # Should be passed to the init_inp argument of an embedding function:
#' init_inp = make_init_inp(prob_perp_bisect(perplexity = 30,
#'                                          input_weight_fn = exp_weight))
#' @family sneer input initializers
#' @export
prob_perp_bisect <- function(perplexity = 30,
                             input_weight_fn = exp_weight,
                             keep_all_results = FALSE,
                             verbose = TRUE) {

  function(inp) {
    d_to_p_result <- d_to_p_perp_bisect(inp$dm, perplexity = perplexity,
                                        weight_fn = input_weight_fn,
                                        verbose = verbose)

    if (keep_all_results) {
      for (name in names(d_to_p_result)) {
        inp[[name]] <- d_to_p_result[[name]]
      }
    }
    else {
      inp$pm <- d_to_p_result$pm
    }
    flush.console()
    inp
  }
}

#' Conditional Probability Matrix from Row Probability Matrix
#'
#' Given a row probability matrix (elements of each row are non-negative and
#' sum to one), this function scales each element by the sum of the matrix so
#' that the elements of the entire matrix sum to one.
#'
#' @param prow Row probability matrix.
#' @return Conditional probability matrix.
prow_to_pcond <- function(prow) {
  clamp(prow / sum(prow))
}

#' Joint Probability Matrix from Row Probability Matrix
#'
#' Given a row probability matrix (elements of each row are non-negative and
#' sum to one), this function scales each element by such that the elements of
#' the entire matrix sum to one, and that the matrix is symmetric, i.e.
#' \code{p[i, j] = p[j, i]}.
#'
#' @param prow Row probability matrix.
#' @return Joint probability matrix.
prow_to_pjoint <- function(prow) {
  clamp(symmetrize_matrix(prow / sum(prow)))
}

#' Symmetric Matrix from Square Matrix
#'
#' The matrix is symmetrized by setting \code{pm[i, j]} and \code{pm[j, i]} to
#' their average, i.e. \code{Pij} = \code{(Pij + Pji)/2} = \code{Pji}.
#'
#' In SSNE and t-SNE, this is used as part of the process of converting the row
#' stochastic matrix of conditional input probabilities to a joint probability
#' matrix.
#'
#' @param pm Square matrix to symmetrize.
#' @return Symmetrized matrix such that \code{pm[i, j]} = \code{pm[j, i]}
symmetrize_matrix <- function(pm) {
  0.5 * (pm + t(pm))
}

#' Input Distance Scaling
#'
#' An initialization method for creating input probabilities.
#'
#' This initialization method scales the input distances so that the average
#' distance is 1. Therefore, if applying other input initializers that use
#' the distances (e.g. to calculate probabilities), this initializer should
#' appear before them in the \code{\link{make_init_inp}} parameter list.
#'
#' @param verbose If \code{TRUE}, information about the scaled distances will be
#' logged.
#' @return Input initialization method for use by \code{\link{make_init_inp}}.
#' @seealso \code{\link{make_init_inp}} for how to use this function as part
#' of an embedding.
#' @examples
#' # Should be passed to the init_inp argument of an embedding function:
#' init_inp = make_init_inp(scale_input_distances())
#'
#' # Should appear before other use of the input distances
#' init_inp = make_init_inp(scale_input_distances(),
#'                          prob_perp_bisect(perplexity = 30,
#'                                          input_weight_fn = exp_weight))
#' @references
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' @family sneer input initializers
#' @export
scale_input_distances <- function(verbose = TRUE) {
  function(inp) {
    inp$dm <- inp$dm / mean(upper_tri(inp$dm))
    if (verbose) {
      summarize(upper_tri(inp$dm), "Scaled inp$dm")
    }
    inp
  }
}
