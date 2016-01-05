# Functions to generate probabilities from coordinates, as used in
# probability-based embedding methods.
# Note on terminology:
# Row Probability Matrix: matrix represents N separate probability distributions,
# where N is the number of rows (or columns). Each row sums to 1. In the context
# of probail
# Conditional Probability Matrx: matrix represents a single probab

#' Probability Matrices
#'
#' For probability-based embedding, sneer works with three types of probability
#' matrix.
#'
#' @section Row Probabilities:
#' Matrix contains N different probability distributions, where N is the number
#' of rows (or columns). Each row sums to one. This structure is called a row
#' stochastic matrix when used in Markov Chains.
#'
#' In the context of embedding, this structure represents a point-based
#' probability: a given entry \code{P[i, j]} should be considered the
#' conditional probability pj|i, the probability that point j is a neighbor of
#' i, given that i has been picked as the reference point.
#'
#' Given the potentially different inhomogeneous distribution of points
#' in a data set, there is therefore no reason that \code{P[i, j]} should be
#' equal to \code{P[j, i]}: the probability that point i is a neigbour of point
#' j is not necessarily the same that j is a neighbor of i.
#'
#' Row probability is the model used in the original (asymmetric) Stochastic
#' Neighbor Embedding paper.
#'
#' In sneer, row probability matrices should be given a \code{type} attribute
#' with value \code{row}.
#'
#' @section Joint Probabilities:
#' Matrix contains 1 probability distribution. The grand sum of the matrix is
#' one.
#'
#' In the context of embedding, this structure represents a pair-based
#' probability: a given entry \code{P[i, j]} should be considered the
#' joint probability pij, the probability that point i and point j are selected
#' as a pair. From this point of view, whether the pair is represented as
#' \code{P[i, j]} or \code{P[j, i]} should be irrelevant. Therefore a joint
#' probability is symmetric.
#'
#' Joint probability is the model used in Symmetric Stochastic Neighbor
#' Embedding and t-Distributed Stochastic Neighbor Embedding. Although both
#' methods use the same techniques and procedures as ASNE, they introduce
#' further steps to process the input probability, which they convert from a
#' row probability form to a joint probability by taking the average of
#' \code{P[i, j]} and \code{P[j, i]} and then renormalizing. The output
#' probabilities don't require the averaging step, because the weight function
#' used in SSNE and t-SNE is symmetric. Only the normalization step is required.
#'
#' In sneer, joint probability matrices should be given a \code{type} attribute
#' with value \code{joint}.
#'
#' @section Conditional Probabilities:
#' An intermediate step between a row probability matrix and a joint probability
#' matrix. The matrix contains 1 probability distribution. The grand sum of the
#' matrix is one.
#'
#' In the context of embedding, this structure represents a pair-based
#' probability: a given entry \code{P[i, j]} should be considered the
#' conditional probability pj|i, the probability that point i and point j are
#' selected as a pair, given that point i was selected first.
#'
#' Unlike the joint probability, there is no restriction that \code{P[i, j]} be
#' equal to \code{P[j, i]}. The question is whether you can justify that making
#' sense in your model of embedding.
#'
#' No embedding I'm aware of uses conditional probabilities directly. However,
#' the output probabilities in SSNE and t-SNE could be thought of as a special
#' case of a conditional matrix: yes, the output \code{P[i, j]} is equal to
#' \code{P[j, i]}, so they're technically joint probabilities, but no special
#' effort was made to make them joint, unlike their counterpart matrices in
#' the input case: those need to be explicitly normalized and then symmetrized.
#' The jointness of the output probabilities in SSNE and t-SNE is entirely a
#' by-product of the symmetric nature of the weighting function used to generate
#' the similarities. In SSNE, if the beta parameter was allowed to vary between
#' points in the output probability, for example, then the resulting matrix
#' would have to be symmetrized to be a joint probability (note also that the
#' stiffness matrix function would have to be rewritten!).
#'
#' In sneer, joint probability matrices should be given a \code{type} attribute
#' with value \code{joint}.
#'
#' @examples
#' \dontrun{
#' # Setting the row probability type
#' prow <- some_func()
#' attr(prow, "type") <- "row"
#'
#' #' # Setting the joint probability type
#' pjoint <- some_other_func()
#' attr(pjoint, "type") <- "joint"
#'
#' # Setting the conditional probability type
#' pcond <- yet_another_func()
#' attr(pcond, "type") <- "cond"
#' }
#' @keywords internal
#' @name probability_matrices
NULL


#' Create Weight Matrix from Output Data
#'
#' Creates a matrix of positive weights from the distance matrix of the
#' embedded coordinates.
#'
#' @param out Output data.
#' @param method Embedding method.
#' @return Weight matrix for the embedded coordinates in \code{out}.
weights <- function(out, method) {
  coords_to_weights(out$ym, method$weight_fn)
}

#' Create Weight Matrix from Coordinates Matrix
#'
#' @param ym Matrix of coordinates.
#' @param weight_fn Function with signature \code{weight_fn(d2m)}
#' where \code{d2m} is a matrix of squared distances between the output
#' coordinates. It should return a weight matrix.
#' @return Weight matrix.
coords_to_weights <- function(ym, weight_fn) {
  d2m <- coords_to_dist2(ym)
  dist2_to_weights(d2m, weight_fn)
}

#' Create Weight Matrix from Squared Distances
#'
#' Weights are subsequently normalized to probabilities in probability-based
#' embedding. This function guarantees that the self-weight is 0.
#'
#' @param d2m Matrix of squared distances.
#' @param weight_fn Function with signature \code{weight_fn(d2m)} where
#' \code{d2m} is a matrix of squared distances.
#' @return Weights matrix. The diagonal (i.e. self-weight) is enforced to be
#' zero.
dist2_to_weights <- function(d2m, weight_fn) {
  wm <- weight_fn(d2m)
  diag(wm) <- 0  # set self-weights to 0
  wm
}

#' Create Row Probability Matrix from Weight Matrix
#'
#' Used in ASNE. The probability matrix is such that all elements are positive
#' and each row sums to 1.
#'
#' @param wm Matrix of weighted distances.
#' @return Row probability matrix.
weights_to_prow <- function(wm) {
  row_sums <- apply(wm, 1, sum)
  pm <- sweep(wm, 1, row_sums, "/")
  pm[is.nan(pm)] <- 1 / ncol(pm)
  clamp(pm)
}

#' Create Conditional Probability Matrix from Weight Matrix
#'
#' Used in SSNE and TSNE. The probability matrix is such that all elements are
#' positive and sum to 1.
#'
#' @param wm Matrix of weighted distances.
#' @return Probability matrix.
weights_to_pcond <- function(wm) {
  qm <- wm / sum(wm)
  clamp(qm)
}

#' Conditional Probability Matrix from Row Probability Matrix
#'
#' Given a row probability matrix (elements of each row are non-negative and
#' sum to one), this function scales each element by the sum of the matrix so
#' that the elements of the entire matrix sum to one.
#'
#' An intermediate step in creating joint probabilities from row probabilities.
#'
#' @param prow Row probability matrix.
#' @return Conditional probability matrix.
prow_to_pcond <- function(prow) {
  prow / sum(prow)
}

#' Convert Row Probability Matrix to a Joint Probability Matrix
#'
#' Given a row probability matrix (elements of each row are non-negative and
#' sum to one), this function scales each element by such that the elements of
#' the entire matrix sum to one, and that the matrix is symmetric, i.e.
#' \code{p[i, j] = p[j, i]}.
#'
#' Used in \code{\link{ssne}}, \code{\link{tsne}} and related methods to convert
#' input probabilities.
#'
#' @param prow Row probability matrix.
#' @return Joint probability matrix.
prow_to_pjoint <- function(prow) {
  symmetrize_matrix(prow_to_pcond(prow))
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

#' Probability Type Conversion
#'
#' Given a probability matrix and an embedding method, this function applies
#' a conversion from the type of the probability matrix into that required
#' by the embedding method.
#'
#' @param pm Probability Matrix. Must have a "type" attribute with one of the
#'   following values:
#'   \describe{
#'   \item{"row"}{A probability where each row sums to one.}
#'   \item{"cond"}{A conditional probablilty matrix where the grand sum of the
#'   entire matrix is 1.}
#'   \item{"joint"}{A joint probability matrix where the grand sum of the entire
#'   matrix is 1, and \code{pm[i, j]} = \code{pm[j, i]} for all pairs i and j.}
#'   }
#'   See \code{\link{probability_matrices}} for more details.
#' @param method Embedding method, which must have a "type" attribute with one
#'   of the values described for \code{pm}. This represents the type of
#'   probability matrix that \code{pm} will be converted into.
#' @return Converted probability matrix.
handle_prob <- function(pm, method) {
  prob_in <- attr(pm, "type")
  if (is.null(prob_in)) {
    stop("P matrix must have type attribute defined")
  }

  prob_out <- method$prob_type
  if (prob_in != prob_out) {
    prob_fn_name <- paste0("p", prob_in, "_to_p", prob_out)
    prob_fn <- get(prob_fn_name)
    if (is.null(prob_fn)) {
      stop("No ", prob_fn_name, " function defined for probability matrix
           conversion")
    }
    pm <- get(prob_fn_name)(pm)
    attr(pm, "type") <- prob_out
  }
  clamp(pm)
}
