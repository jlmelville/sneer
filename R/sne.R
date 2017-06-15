# Functions defining the SNE-family of probability-based embedding methods.

# Probability-Based Embeddings
#
# The available embedding methods which work by reproducing probabilities
# based on the input distances, rather than the input distances directly.
#
# @examples
# \dontrun{
# embed_prob(method = tsne(), ...)
# embed_prob(method = asne(), ...)
# }
# @keywords internal
# @name probability_embedding_methods
# @family sneer probability embedding methods
NULL

# Asymmetric Stochastic Neighbor Embedding (ASNE)
#
# A probability-based embedding method.
#
# The original SNE method, this uses exponential weighting for both the input
# and output probabilities. Unlike \code{ssne} and \code{tsne},
# the probabilities are defined with respect to points, not pairs: an element
# Pij from the NxN probability matrix P should be thought of as a conditional
# probabilty, pj|i, the probability that point j would be chosen as a close
# neighbor of point i.
#
# The probability matrix used in ASNE:
#
# \itemize{
#  \item{represents one N row-wise probability distributions, where N is the
#  number of points in the data set, i.e. the row sums of the matrix are all
#   one.}
#  \item{is asymmetric, i.e. there is no requirement that
#  \code{p[i, j] == p[j, i]}.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Row probability matrix based on embedded coordinates.}
# }
#
# @param beta Precision-like parameter of the exponential similarity kernel
#  function: the larger it is, the steeper the curve. This is usually left at
#  its default value of 1.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Hinton, G. E., & Roweis, S. T. (2002).
# Stochastic neighbor embedding.
# In \emph{Advances in neural information processing systems} (pp. 833-840).
# @seealso ASNE uses the \code{kl_cost} cost function and the
#   \code{exp_weight} similarity function. The return value of this
#   function should be used with the \code{embed_prob} embedding
#   function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# embed_prob(method = asne(), ...)
# }
asne <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  list(
    cost = kl_fg(),
    kernel = exp_kernel(beta = beta),
    stiffness_fn = function(method, inp, out) {
      asne_stiffness(inp$pm, out$qm, beta = method$kernel$beta)
    },
    out_keep = c("qm"),
    prob_type = "row",
    eps = eps,
    verbose = verbose
  )
}

# Symmetric Stochastic Neighbor Embedding (SSNE)
#
# A probability-based embedding method.
#
# SSNE is a variant of \code{asne} where the probabilities are with
# respect to pairs of points, not individual points. The element
# \code{P[i, j]} in matrix P should be thought of as the probability of
# selecting a pair of points i and j as close neighbors. As a result, unlike
# ASNE, the probability matrix in SSNE:
#
# \itemize{
#  \item{represents one probability distribution, i.e. the grand sum of the
#  matrix is one.}
#  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#  probabilities are joint probabilities.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Joint probability matrix based on embedded coordinates.}
# }
#
# @param beta Precision-like parameter of the exponential similarity kernel
#  function: the larger it is, the steeper the curve. This is usually left at
#  its default value of 1.
# @param eps Small floating point value used to prevent numerical problems,
#  e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Cook, J., Sutskever, I., Mnih, A., & Hinton, G. E. (2007).
# Visualizing similarity data with a mixture of maps.
# In \emph{International Conference on Artificial Intelligence and Statistics} (pp. 67-74).
#
# Van der Maaten, L., & Hinton, G. (2008).
# Visualizing data using t-SNE.
# \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
# @seealso SSNE uses the \code{kl_cost} cost function and the
#   \code{exp_weight} similarity function. The return value of this
#   function should be used with the \code{embed_prob} embedding
#   function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# embed_prob(method = ssne(), ...)
# }
ssne <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    asne(beta = beta, eps = eps, verbose = verbose),
    stiffness_fn = function(method, inp, out) {
      ssne_stiffness(inp$pm, out$qm, beta = method$kernel$beta)
    },
    prob_type = "joint")
}

# t-Distributed Stochastic Neighbor Embedding (t-SNE).
#
# A probability-based embedding method.
#
# t-SNE is a variant of \code{ssne} where the similarity function used
# to generate output probabilities is the Student t-Distribution with one
# degree of freedom.
#
# The probability matrix in t-SNE:
#
# \itemize{
#  \item{represents one probability distribution, i.e. the grand sum of the
#  matrix is one.}
#  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#  probabilities are joint probabilities.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#  \code{wm}.}
#  \item{\code{wm}}{Weight matrix generated from the distances between points
#  in \code{ym}.}
# }
#
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Van der Maaten, L., & Hinton, G. (2008).
# Visualizing data using t-SNE.
# \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
# @seealso t-SNE uses the \code{kl_cost} cost function and the
#   \code{tdist_weight} similarity function. The return value of this
#   function should be used with the \code{embed_prob} embedding
#   function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# embed_prob(method = tsne(), ...)
# }
tsne <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    ssne(eps = eps, verbose = verbose),
    kernel = tdist_kernel(),
    stiffness_fn = function(method, inp, out) {
      tsne_stiffness(inp$pm, out$qm, out$wm)
    },
    out_keep = c("qm", "wm")
  )
}

# t-distributed Asymmetric Stochastic Neighbor Embedding (t-ASNE)
#
# A probability-based embedding method.
#
# Creates a list of functions that collectively implement t-ASNE: a method that
# I just made up to illustrate how to explore different aspects of embedding
# within sneer: this uses the t-distributed distance weighting of t-SNE, but
# for probability generation uses the point-wise distribution of ASNE.
#
# The probability matrix used in t-ASNE:
#
# \itemize{
#  \item{represents one N row-wise probability distributions, where N is the
#  number of points in the data set, i.e. the row sums of the matrix are all
#   one.}
#  \item{is asymmetric, i.e. there is no requirement that
#  \code{p[i, j] == p[j, i]}.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Row probability matrix based on the weight matrix
#  \code{wm}.}
#  \item{\code{wm}}{Weight matrix generated from the distances between points
#  in \code{ym}.}
# }
#
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso t-ASNE uses the \code{kl_cost} cost function and the
#   \code{kl_cost} similarity function. The return value of this
#   function should be used with the \code{embed_prob} embedding
#   function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# embed_prob(method = tasne(), ...)
# }
tasne <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    tsne(eps = eps, verbose = verbose),
    stiffness_fn = function(method, inp, out) {
      tasne_stiffness(inp$pm, out$qm, out$wm)
    },
    prob_type = "row")
}

# t-Distributed "Pairwise" Stochastic Neighbor Embedding (t-PSNE)
#
# A probability-based embedding method.
#
# Identical to t-SNE, except the input probability matrix is not symmetrized.
# If you compare the embedding produced by this method and those of t-ASNE
# and t-SNE, you can evaluate the relative impact of defining pair-wise
# probabilities versus point-wise probabilities, and that of enforcing that
# the pair-wise probabilities are symmetric.
#
# The probability matrix used in t-PSNE:
#
# \itemize{
#  \item{represents one probability distribution, i.e. the grand sum of the
#  matrix is one.}
#  \item{is asymmetric, i.e. there is no requirement that
#  \code{p[i, j] == p[j, i]}.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Row probability matrix based on the weight matrix
#  \code{wm}.}
#  \item{\code{wm}}{Weight matrix generated from the distances between points
#  in \code{ym}.}
# }
tpsne <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    tasne(eps = eps, verbose = verbose),
    prob_type = "cond")
}

# Heavy-Tailed Symmetric Stochastic Neighbor Embedding (HSSNE)
#
# A probability-based embedding method.
#
# HSSNE is a generalization of \code{ssne} and \code{tsne},
# which uses the \code{heavy_tail_weight} similarity function to
# generate its probabilities.
#
# The heavy tailedness of the weighting function is controlled by the parameter
# \eqn{\alpha}{alpha}, As \eqn{\alpha \to 0}{alpha approaches 0}, the weighting
# function becomes exponential (like SSNE). At \eqn{\alpha = 1}{alpha = 1},
# the weighting function is the t-distribution with one degree of freedom
# (t-SNE). Increasing \eqn{\alpha}{alpha} further will take you into the realm
# of functions with even heavier tails.
#
# Additionally, HSSNE allows control over \eqn{\beta}{beta}, the degree of
# precision (inverse of the bandwidth) of the function. Normally, this set to one
# for the output distances in t-SNE and related methods.
#
# The probability matrix used in HSSNE:
#
# \itemize{
#  \item{represents one probability distribution, i.e. the grand sum of the
#  matrix is one.}
#  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#  probabilities are joint probabilities.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#  \code{wm}.}
#  \item{\code{wm}}{Weight matrix generated from the distances between points
#  in \code{ym}.}
# }
#
# @param alpha Tail heaviness of the kernel similarity function. Must be
# greater than zero. When set to a small value this method is equivalent to
# SSNE. When set to one to one, this method behaves like t-SNE.
# @param beta The precision of the kernel similarity function. Becomes
# equivalent to half the precision in the Gaussian distribution of distances as
# \code{alpha} approaches zero. Leave at the default value of 1 if you want
# consistency with SSNE and ASNE.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
# Heavy-tailed symmetric stochastic neighbor embedding.
# In \emph{Advances in neural information processing systems} (pp. 2169-2177).
# @seealso HSSNE uses the \code{heavy_tail_weight} similarity function.
# The return value of this function should be used with the
# \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# # behave like SSNE
# embed_prob(method = hssne(alpha = 0), ...)
#
# # behave like t-SNE
# embed_prob(method = hssne(alpha = 1), ...)
# }
hssne <- function(beta = 1, alpha = 0, eps = .Machine$double.eps,
                  verbose = TRUE) {
  lreplace(
    tsne(eps = eps, verbose = verbose),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha),
    stiffness_fn = function(method, inp, out) {
      hssne_stiffness(inp$pm, out$qm, out$wm, alpha = method$kernel$alpha,
                      beta = method$kernel$beta)
    }
  )
}

# "Reverse" Asymmetric Stochastic Neighbor Embedding (RASNE)
#
# A probability-based embedding method.
#
# Like \code{asne}, but with the cost function using the "reverse" form
# of the Kullback-Leibler divergence, i.e. KL(Q||P).
#
# The probability matrix in RASNE:
#
# \itemize{
#  \item{represents one N row-wise probability distributions, where N is the
#  number of points in the data set, i.e. the row sums of the matrix are all
#   one.}
#  \item{is asymmetric, i.e. there is no requirement that
#  \code{p[i, j] == p[j, i]}.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Row probability matrix based on embedded coordinates.}
# }
#
# @param beta Precision-like parameter of the exponential similarity kernel
#  function: the larger it is, the steeper the curve. This is usually left at
#  its default value of 1.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso RASNE uses the \code{kl_cost} cost function and the
#   \code{exp_weight} similarity function. The return value of this
#   function should be used with the \code{embed_prob} embedding
#   function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# embed_prob(method = asne(), ...)
# }
rasne <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    asne(beta = beta, eps = eps, verbose = verbose),
    cost = reverse_kl_fg(),
    stiffness_fn = function(method, inp, out) {
      reverse_asne_stiffness(inp$pm, out$qm, out$rev_kl,
                             beta = method$kernel$beta, eps = method$eps)
    },
    out_updated_fn = klqp_update)
}

# "Reverse" Symmetric Stochastic Neighbor Embedding (RSSNE)
#
# A probability-based embedding method.
#
# Like \code{ssne}, but with the cost function using the "reverse" form
# of the Kullback-Leibler divergence, i.e. KL(Q||P).
#
# The probability matrix in RSSNE:
#
# \itemize{
#  \item{represents one probability distribution, i.e. the grand sum of the
#  matrix is one.}
#  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#  probabilities are joint probabilities.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Joint probability matrix based on embedded coordinates.}
# }
#
# @param beta Precision-like parameter of the exponential similarity kernel
#  function: the larger it is, the steeper the curve. This is usually left at
#  its default value of 1.
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso SSNE uses the \code{reverse_kl_cost} cost function and the
#   \code{exp_weight} similarity function. The return value of this
#   function should be used with the \code{embed_prob} embedding
#   function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# embed_prob(method = rssne(), ...)
# }
rssne <- function(beta = 1, eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    rasne(beta = beta, eps = eps, verbose = verbose),
    stiffness_fn = function(method, inp, out) {
      reverse_ssne_stiffness(inp$pm, out$qm, out$rev_kl,
                             beta = method$kernel$beta, eps = method$eps)
    },
    prob_type = "joint")
}

# "Reverse" t-Distributed Stochastic Neighbor Embedding (RTSNE).
#
# A probability-based embedding method.
#
# Like \code{tsne}, but with the cost function using the "reverse" form
# of the Kullback-Leibler divergence, i.e. KL(Q||P).
#
# The probability matrix in RTSNE:
#
# \itemize{
#  \item{represents one probability distribution, i.e. the grand sum of the
#  matrix is one.}
#  \item{is symmetric, i.e. \code{P[i, j] == P[j, i]} and therefore the
#  probabilities are joint probabilities.}
# }
#
# @section Output Data:
# If used in an embedding, the output data list will contain:
# \describe{
#  \item{\code{ym}}{Embedded coordinates.}
#  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#  \code{wm}.}
#  \item{\code{wm}}{Weight matrix generated from the distances between points
#  in \code{ym}.}
# }
#
# @param eps Small floating point value used to prevent numerical problems,
# e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso RTSNE uses the \code{reverse_kl_cost} cost function and the
#   \code{tdist_weight} similarity function. The return value of this
#   function should be used with the \code{embed_prob} embedding
#   function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# embed_prob(method = tsne(), ...)
# }
rtsne <- function(eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    rssne(eps = eps, verbose = verbose),
    kernel = tdist_kernel(),
    stiffness_fn = function(method, inp, out) {
      reverse_tsne_stiffness(inp$pm, out$qm, out$wm, out$rev_kl,
                             eps = method$eps)
    },
    out_keep = c("qm", "wm"))
}
