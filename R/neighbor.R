# Neighborhood Retrieval Metrics

#' Area Under the RNX Curve
#'
#' The RNX curve is formed by calculating the \code{\link{rnx_crm}} metric for
#' different sizes of neighborhood. Each value of RNX is scaled according to
#' the natural log of the neighborhood size, to give a higher weight to smaller
#' neighborhoods. An AUC of 1 indicates perfect neighborhood preservation, an
#' AUC of 0 is due to random results.
#'
#' @note Calculating the RNX curve requires calculating a co-ranking matrix,
#'  which is not a very efficient operation, because it requires iterating
#'  over every element of the distance matrix, the size of which scales with
#'  the square of the number of observations. Be sure you really want to
#'  calculate this for datasets with more than approximately 1,000 observations.
#'
#' @param inp Input data. The input distance matrix will be calculated if it's
#' not present.
#' @param out Output data. The output distance matrix will be calculated if
#' it's not present.
#' @return Area under the RNX curve.
#' @references
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
rnx_auc <- function(inp, out) {
  if (is.null(inp$dm)) {
    inp$dm <- distance_matrix(inp$xm)
  }
  if (is.null(out$dm)) {
    out$dm <- distance_matrix(out$ym)
  }
  list(name = "rnx_auc", value = rnx_auc_dm(inp$dm, out$dm))
}

#' Area Under the RNX Curve
#'
#' The RNX curve is formed by calculating the \code{\link{rnx_crm}} metric for
#' different sizes of neighborhood. Each value of RNX is scaled according to
#' the natural log of the neighborhood size, to give a higher weight to smaller
#' neighborhoods. An AUC of 1 indicates perfect neighborhood preservation, an
#' AUC of 0 is due to random results.
#'
#' @param din Input distance matrix.
#' @param dout Output distance matrix.
#' @return Area under the RNX curve.
#' @references
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
rnx_auc_dm <- function(din, dout) {
  rnx_auc_crm(coranking_matrix(din, dout))
}

#' Co-ranking Matrix
#'
#' Calculates the co-ranking matrix for an embedding.
#'
#' The co-ranking matrix is the basic data structure used for calculating
#' various quality metrics, such as \code{\link{qnx_crm}},
#' \code{\link{rnx_crm}} and \code{\link{bnx_crm}}.
#'
#' The co-ranking matrix is an N x N matrix where N is the number of
#' observations (so it's the same size as the distance matrices its
#' constructed from). The element (i, j) in the co-ranking matrix is the
#' number of times an ith-nearest neighbor of an observation in the input
#' distance matrix was the jth-nearest neighbor in the output space.
#'
#' The lower diagonal represents "intrusions". This is when observations
#' have a larger rank in the input space than in the output space,
#' i.e. non-neighbors are falsely marked as neighbors in the output space.
#'
#' The upper diagonal represents "extrusions". This occurs when observations
#' have a smaller rank in the input space than in the output space,
#' i.e. true neighbors are falsely marked as non-neighbors in the output space.
#'
#' @param din Input distance matrix.
#' @param dout Output distance matrix.
#' @return Co-ranking matrix.
#' @references
#' Lee, J. A., & Verleysen, M. (2009).
#' Quality assessment of dimensionality reduction: Rank-based criteria.
#' \emph{Neurocomputing}, \emph{72(7)}, 1431-1443.
coranking_matrix <- function(din, dout) {
  crm <- matrix(0, nrow = nrow(din), ncol = ncol(dout))
  for (i in 1:nrow(din)) {
    rin <- rank(din[i,])
    rout <- rank(dout[i,])
    for (j in 1:length(rin)) {
      crm[rin[j], rout[j]] <- crm[rin[j], rout[j]] + 1
    }
  }
  crm
}

#' Area Under the RNX Curve
#'
#' The RNX curve is formed by calculating the \code{\link{rnx_crm}} metric for
#' different sizes of neighborhood. Each value of RNX is scaled according to
#' the natural log of the neighborhood size, to give a higher weight to smaller
#' neighborhoods. An AUC of 1 indicates perfect neighborhood preservation, an
#' AUC of 0 is due to random results.
#'
#' @param crm Co-ranking matrix.
#' @return Area under the curve.
#' @references
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
rnx_auc_crm <- function(crm) {
  n <- nrow(crm)
  num <- 0
  den <- 0
  for (k in 1:(n - 2)) {

    num <- num + rnx_crm(crm, k) / k
    den <- den + (1 / k)
  }
  num / den
}

#' Rescaled Agreement Between K-ary Neighborhoods (RNX)
#'
#' RNX is a scaled version of QNX which measures the agreement between two
#' embeddings in terms of the shared number of k-nearest neighbors for each
#' observation. RNX gives a value of 1 if the neighbors are all preserved
#' perfectly and a value of 0 for a random embedding.
#'
#' @param crm Co-ranking matrix. Create from a pair of distance matrices with
#' \code{\link{coranking_matrix}}.
#' @param k Neighborhood size.
#' @return RNX for \code{k}.
#' @references
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
rnx_crm <- function(crm, k) {
  n <- nrow(crm)
  ((qnx_crm(crm, k) * (n - 1)) - k) / (n - 1 - k)
}

#' Average Normalized Agreement Between K-ary Neighborhoods (QNX)
#'
#' QNX measures the degree to which an embedding preserves the local
#' neighborhood around each observation. For a value of K, the K closest
#' neighbors of each observation are retrieved in the input and output space.
#' For each observation, the number of shared neighbors can vary between 0
#' and K. QNX is simply the average value of the number of shared neighbors,
#' normalized by K, so that if the neighborhoods are perfectly preserved, QNX
#' is 1, and if there is no neighborhood preservation, QNX is 0.
#'
#' For a random embedding, the expected value of QNX is approximately
#' K / (N - 1) where N is the number of observations. Using RNX
#' (\code{\link{rnx_crm}}) removes this dependency on K and the number of
#' observations.
#'
#' @param crm Co-ranking matrix. Create from a pair of distance matrices with
#' \code{\link{coranking_matrix}}.
#' @param k Neighborhood size.
#' @return QNX for \code{k}.
#' @references
#' Lee, J. A., & Verleysen, M. (2009).
#' Quality assessment of dimensionality reduction: Rank-based criteria.
#' \emph{Neurocomputing}, \emph{72(7)}, 1431-1443.
qnx_crm <- function(crm, k) {
  sum(crm[1:k, 1:k]) / (k * nrow(crm))
}


#' Intrusions and Extrusions for K-ary Neighborhoods (BNX)
#'
#' BNX measures the degree of intrusions versus extrusions that contributes
#' to the QNX measure of embedding error. If BNX > 0 this means that intrusions
#' dominate over extrusions: i.e. non-neighbors in the input space are neighbors
#' in the output space. BNX < 0 means that extrusions dominate over intrusions:
#' neighbors in the input space tend to be non-neighbors in the output space.
#'
#' @param crm Co-ranking matrix. Create from a pair of distance matrices with
#' \code{\link{coranking_matrix}}.
#' @param k Neighborhood size.
#' @return BNX for \code{k}.
bnx_crm <- function(crm, k) {
  kcrm <- crm[1:k, 1:k]
  intrusions <- sum(kcrm[lower.tri(kcrm)])
  extrusions <- sum(kcrm[upper.tri(kcrm)])
  (intrusions - extrusions) / (k * nrow(crm))
}

#' Indexes of the k-largest numbers.
#'
#' Given a vector of numbers, return the indexes of the k-largest
#' values.
#'
#' @param x Vector of numbers.
#' @param k Top k results to return
#' @return Vector of the indexes of the \code{k} largest values in \code{x}.
k_largest_ind <- function(x, k) {
  which(k >= sort(k, decreasing = TRUE)[k], arr.ind = TRUE)
}

#' Indexes of the k-smallest numbers in a vector.
#'
#' Given a vector of numbers, return the indexes of the k-smallest
#' values.
#'
#' @param x Vector of numbers.
#' @param k Top k results to return
#' @return Vector of the indexes of the \code{k} smallest values in \code{x}.
k_smallest_ind <- function(x, k) {
  k_largest_ind(-x, k)
}

#' Indexes of the shared neighbors between two distance vectors
#'
#' Return the indexes of shared k-closest neighbors in two lists of distances.
#'
#' @param di list of distances
#' @param dj list of distances
#' @param k The size of the shared neighborhood
#' @return Vector of the indexes of the elements which are among both the
#' \code{k}-smallest values of \code{di} and the \code{k}-smallest
#' values of \code{dj}.
k_shared_nbrs_ind <- function(di, dj, k) {
  nindi <- k_smallest_ind(di, k)
  nindj <- k_smallest_ind(dj, k)

  Reduce(intersect, list(nindi, nindj))
}

#' Neighborhood Preservation
#'
#' For the K nearest neighbors in one set of distances, returns the number of
#' those neighbors which are also K nearest neighbors in another list,
#' normalized with respect to K.
#'
#' The neighborhood preservation can vary between 0 (no neighbors in common)
#' and 1 (perfect preservation). However, random performance gives an
#' approximate value of K / K - 1.
#'
#' @param di Vector of distances.
#' @param dj Vector of distances.
#' @param k Size of the neighborhood to consider.
#' @return The number of shared neighbors in the equivalent neighbor lists of
#' \code{di} and \code{dj}.
nbr_pres_i <- function(di, dj, k) {
  length(k_shared_nbrs_ind(di, dj, k)) / k
}

#' Neighborhood Preservation Between Distance Matrices
#'
#' Calculates the neighborhood preservation for each observation in a dataset,
#' represented by two distance matrices. The first matrix is the "ground truth",
#' the second being the estimation or approximation.
#' The neighborhood preservation is calculated for each row where each element
#' d[i, j] is taken to be the distance between observation i and j.
#'
#' The neighborhood preservation can vary between 0 (no neighbors in common)
#' and 1 (perfect preservation). However, random performance gives an
#' approximate value of K / K - 1.
#'
#' @note This is not a very efficient way to calculate the preservation if you
#'  want to calculate the value for multiple values of \code{k}. For more
#'  global measures of preservation, see \code{\link{rnx_auc}}.
#'
#' @param din Distance matrix. The "ground truth" or reference distances.
#' @param dout Distance matrix. A set of distances to compare to the reference
#'  distances.
#' @param k The size of the neighborhood, where k is the number of neighbors to
#'  include in the neighborhood.
#' @return Vector of preservation values, one for each row of the distance
#'  matrix.
#' @export
nbr_pres <- function(din, dout, k) {
  preservations <- vector(mode = "numeric", length = nrow(din))
  for (i in 1:nrow(din)) {
    preservations[i] <- nbr_pres_i(din[i,], dout[i,], k)
  }
  preservations
}

