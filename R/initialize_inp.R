# Functions for initializing the input data.

#' Create an input initialization callback.
#'
#' Factory function for input initialization callbacks. During input
#' initialization, the high dimensional distances or coordinates will be used
#' to generate the matrices necessary for the embedding, e.g. input distances
#' and probabilities.
#'
#' The callback that is returned generates an input probability according
#' to the method described in the TSNE paper: for each point in the input
#' data, a probability distribution is generated with respect to all the
#' other points such that the perplexity of the distribution is a user-defined
#' value. This probability is suitable for use in the ASNE method. For SSNE
#' and TSNE a further processing stage is required to create a joint probability
#' over all pairs of points. This can be done in the \code{after_init} callback
#' available to the embedding methods.
#'
#' @param perplexity Target perplexity value for the probability distributions.
#' @param input_weight_fn Weighting function for distances. It should have the
#' signature \code{input_weight_fn(d2m, beta)}, where \code{d2m} is a matrix of
#' squared distances and \code{beta} is a real-valued scalar parameter which
#' will be varied as part of the search to produce the desired
#' \code{perplexity}. The function should return a matrix of weights
#' corresponding to the transformed squared distances passed in as arguments.
#' @param keep_all_results If \code{true} then the list returned by the callback
#' will also contain a vector of \code{beta} parameters that generated the
#' probability matrix. Otherwise, only the probability matrix is returned.
#' @return A callback function with signature \code{fn(xm)} where \code{xm}
#' is the input coordinates or a distance matrix. This function will return
#' a list containing: \itemize{
#'  \item \code{xm} Input coordinates if these were provided.
#'  \item \code{dm} Input distances.
#'  \item \code{pm} Input probabilities.
#'  \item \code{beta} Input weighting parameters that produced the
#'  probabilities. Only provided if \code{keep_all_results} is \code{TRUE}.
#'}
#' @seealso \code{\link{embed_sim}} for how to use this function for configuring
#' an embedding.
#' @examples
#' # Set target perplexity of each probability distribution to 30
#' make_init_inp(perplexity = 30)
#'
#' # Set target perplexity of each probability distribution to 30 but use
#' # a different weighting function.
#' make_init_inp(perplexity = 30, input_weight_fn = sqrt_exp_weight)
#'
#' # Perplexity of 50, and keep the values of the exponential parameter for
#' # later processing or reporting.
#' make_init_inp(perplexity = 50, keep_all_result = TRUE)
#'
#' # Should be passed to the init_inp argument of an embedding function:
#' \dontrun{
#'  embed_sim(init_inp = make_init_inp(perplexity = 30,
#'                                     input_weight_fn = exp_weight), ...)
#' }
make_init_inp <- function(perplexity = 30,
                          input_weight_fn = exp_weight,
                          keep_all_results = FALSE,
                          verbose = TRUE) {
  function(xm) {
    inp <- list()
    if (class(xm) == "dist") {
      inp$dm <- clamp(as.matrix(xm))
    } else {
      inp$xm <- as.matrix(xm)
      inp$dm <- distance_matrix(inp$xm)
    }

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

    inp
  }
}

#' Convert a row probability matrix to a conditional probability matrix.
#'
#' Given a row probability matrix (elements of each row are non-negative and
#' sum to one), this function scales each element by the sum of the matrix so
#' that the elements of the entire matrix sum to one.
#'
#' @param prow Row probability matrix.
#' @return Conditional probability matrix.
prow_to_pcond <- function(prow) {
  clamp(prow/sum(prow))
}

#' Convert a row probability matrix to a join probability matrix.
#'
#' Given a row probability matrix (elements of each row are non-negative and
#' sum to one), this function scales each element by such that the elements of
#' the entire matrix sum to one, and that the matrix is symmetric, i.e.
#' \code{p[i, j] = p[j, i]}.
#'
#' @param prow Row probability matrix.
#' @return Joint probability matrix.
prow_to_pjoint <- function(prow) {
  clamp(symmetrize_matrix(prow/sum(prow)))
}

#' Create a symmetric matrix from a square matrix.
#'
#' The matrix is symmetrized by setting \code{pm[i, j]} and \code{pm[j, i]} to
#' their average, i.e. \code{Pij} = \code{(Pij + Pji)/2} = \code{Pji}.
#'
#' In SSNE and TSNE, this is used as part of the process of converting the row
#' stochastic matrix of conditional input probabilities to a joint probability
#' matrix.
#'
#' @param pm Square matrix to symmetrize.
#' @return Symmetrized matrix such that \code{pm[i, j]} = \code{pm[j, i]}
symmetrize_matrix <- function(pm) {
  0.5 * (pm + t(pm))
}
