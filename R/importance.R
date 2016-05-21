# Code to implement w-SSNE as described by Yang and co-workers.

# Degree Centrality
#
# Calculates the degree centrality of a probability matrix.
#
# The input probability matrix is weighted to create the symmetrized nearest
# neighbor graph. The degree centrality of node i is then the sum of row
# and column i, where the element \code{m[i, j]} of matrix \code{m} can be
# interpreted as the value of the weighted edge from point i to j.
#
# @param inp Input data
# @param out Output data
# @param method Embedding method
# @return list containing the updated method.
#
# @references
# Yang, Z., Peltonen, J., & Kaski, S. (2014).
# Optimization equivalence of divergences improves neighbor embedding.
# In \emph{Proceedings of the 31st International Conference on Machine Learning (ICML-14)}
# (pp. 460-468).
calculate_importance <- function(inp, out, method) {

  deg <- centrality(inp, method)
  if (method$verbose) {
    summarize(deg, "centrality")
  }

  method$kernel$im <- outer(deg, deg)
  inp$deg <- deg
  list(method = method, inp = inp)
}

# Indegree Centrality
#
# A node importance measure for directed graphs.
#
# Sums the (possibly weighted) edges that are directed towards each node. For
# probability matrices, column i is interpreted as containing the weighted
# edges that are directed to node i.
#
# @param m Nearest neighbor graph matrix.
# @return Indegree centrality of \code{m}.
indegree_centrality <- function(m) {
  colSums(m)
}

# Outdegree Centrality
#
# A node centrality measure for graphs.
#
# Sums the (possibly weighted) edges that are directed from each node. For
# probability matrices, row i is interpreted as containing the weighted
# edges that are directed from node i.
#
# @param m Nearest neighbor graph matrix.
# @return Outdegree centrality of \code{m}.
outdegree_centrality <- function(m) {
  rowSums(m)
}

# Degree Centrality
#
# A node centrality measure for graphs.
#
# Sums the (possibly weighted) edges that are incident to each node. For
# probability matrices, row and column i is interpreted as containing the
# weighted edges that are connected to and from node i.
#
# @param m Nearest neighbor graph matrix.
# @return Degree centrality of \code{m}.
degree_centrality <- function(m) {
  indegree_centrality(m) + outdegree_centrality(m)
}

# Convert an Embedding Method to a Weighted Version
#
# Modifies the similarity kernel of an embedding method so that each
# weight is further multiplied by the importance of the two observations the
# weight was based on.
#
# The new weight matrix, which we'll call the "importance" weight matrix, is
# calculated as
#
# \deqn{W_{imp} = WM}{W_imp = W*M}
#
# where W is the original weight matrix and M is the importance matrix, which
# is the outer product of an importance vector d with the same length as the
# number of observations in the dataset, i.e.:
#
# \deqn{M_{ij} = d_{i}d_{j}}{M_ij = di * dj}
#
# In the originating paper by Yang and co-workers, they suggest using the
# degree centrality of each observation as the importance. This involves
# interpreting the input probability matrix as a weighted adjacency matrix,
# i.e. treat the dataset as a fully connected graph, with weighted edges, where
# the weight for the edge between observation i and j is the input probability,
# p_ij.
#
# @param method Embedding method to convert into an importance weighted
# version.
# @param centrality_fn The importance function. By default, the
# degree centrality. Must be a function with the signature
# \code{centrality_fn(m)} where \code{m} is the nearest neighbour graph based
# on the input probabilities, and returning a vector with the same length
# as the number of rows of the matrix, where the ith element represent the
# centrality measure of the ith observation.
# @return Converted embedding method.
#
# @references
# Yang, Z., Peltonen, J., & Kaski, S. (2014).
# Optimization equivalence of divergences improves neighbor embedding.
# In \emph{Proceedings of the 31st International Conference on Machine Learning (ICML-14)}
# (pp. 460-468).
importance_weight <- function(method, centrality_fn = degree_centrality) {
  method$kernel <- imp_kernel(method$kernel)
  method$centrality_fn <- centrality_fn
  method <- on_inp_updated(method, calculate_importance)$method
  method
}

# Convert an existing similarity kernel to an importance weighted version.
#
# The original kernel is used to produce the weight matrix (or the derivative)
# as usual, then multipled by the importance matrix. This should be stored
# as \code{kernel$im}.
#
# @param kernel Similarity kernel to convert.
# @return Kernel with importance weighting.
imp_kernel <- function(kernel) {
  kernel$orig_fn <- kernel$fn
  kernel$orig_gr <- kernel$gr
  kernel$fn <- function(kernel, d2m) {
    kernel$orig_fn(kernel, d2m) * kernel$im
  }
  attr(kernel$fn, "type") <- attr(kernel$orig_fn, "type")
  kernel$gr <- function(kernel, d2m) {
    kernel$orig_gr(kernel, d2m) * kernel$im
  }

  kernel
}

# Convert a Probability Matrix to a Nearest Neighbor Adjacency Graph.
#
# Weights a probability matrix so that it approximates a (possibly symmetrized)
# nearest neighbor graph. In the context of probability-based embedding, the
# nearest neighbor adjancency graph is the same as the weight matrix that would
# result if a step function was used, rather than the usual exponential
# weighting. Given a row probability matrix we can then map back to the nearest
# neighbor graph by multiplying each probability by the number of neighbours
# required, which is equivalent to the perplexity.
#
# If the matrix is already normalized and symmetrized then the resulting
# neighbor graph is also symmetrized.
#
# @param m Probability matrix.
# @param k The number of neighbors.
# @return Nearest neighbor graph.
nn_graph <- function(m, k) {
  # this weighting ensures we get the correct (symmetrized) NN graph
  # whether the probabilities are row-based or matrix-based
  m * k * (nrow(m) / sum(m))
}

# Centrality Measure for Input Probability
#
# Converts the input probability to a nearest neighbour adjacency matrix
# and then calculates a centrality measure.
#
# @param inp Input method
# @param method Embedding method.
# @return Vector containing the centrality measure for each observation in the
# data set.
centrality <- function(inp, method) {
  m <- nn_graph(inp$pm, inp$perp)
  if (is.null(method$centrality_fn)) {
    method$centrality_fn <- degree_centrality
  }
  method$centrality_fn(m)
}

