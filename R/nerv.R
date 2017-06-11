# Neighbor Retrieval Visualizer (NeRV)
#
# A probability-based embedding method.
#
# NeRV is a variant of Asymmetric Stochastic Neighbor Embedding
# (see \code{asne}), with a modified cost function: in addition to
# calculating the Kullback-Leibler divergence of the output probabilities Q,
# from the input probabilities, P, it also includes the divergence of P from Q.
# The final cost function is a weighted sum of these two individual functions.
# Hence ASNE is a special case of NeRV where all the weight is placed on the
# first component of the cost function.
#
# From an information retrieval perspective, the weighting factor allows the
# user to place a relative weight on false positives: points on the embedded
# map which have a close distance, but a low input probability, i.e. should not
# have been embedded as close neighbors, versus false negatives: pairs with a
# large distance in the output coordinates, but a high input probability, i.e.
# should have been embedded as close neighbors. From this perspective, ASNE
# is the equivalent of emphasising false positives over false negatives.
#
# Additionally, where ASNE uses an exponential function with a parameter set
# to 1 for all pairs of points for its output weighting function, NeRV uses
# the parameters calculated from the input probability matrix which can (and
# do) vary for each observation in the data set. For a more direct compairson
# with ASNE, use the uniform NeRV method \code{unerv} which uses one parameter
# for all weight generation in the output.
#
# The parameter associated with the exponential kernel is sometimes referred
# to as the "precision" in the literature (and in other parts of the help
# text). NeRV already uses the term "precision" as defined in terms of
# information retrieval, so when referring to the output weighting kernel
# function, what's called the "precision" in other parts of the documentation
# is just called the kernel parameter (or "bandwidth") when discussing NeRV.
#
# The probability matrix used in NeRV:
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
#  \item{\code{qm}}{Joint probability matrix based on the weight matrix
#  \code{wm}.}
# }
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to ASNE. Must be a value between 0 and 1.
# @param eps Small floating point value used to prevent numerical problems,
#   e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
# Information retrieval perspective to nonlinear dimensionality reduction for
# data visualization.
# \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
# @seealso NeRV uses the \code{nerv_cost} cost function and the
#   \code{exp_weight} similarity function for converting distances to
#   probabilities.
# The return value of this function should be used with the
# \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# # default NeRV settings
# embed_prob(method = nerv(lambda = 0.5), ...)
#
# # equivalent to ASNE or emphasis on recall over precision
# embed_prob(method = nerv(lambda = 1), ...)
#
# # puts emphasis on precision over recall
# embed_prob(method = nerv(lambda = 0), ...)
# }
nerv <- function(lambda = 0.5, eps = .Machine$double.eps, verbose = TRUE) {
  method <- lreplace(
    asne(eps = eps, verbose = verbose),
    cost = nerv_fg(lambda = lambda),
    stiffness_fn = function(method, inp, out) {
      nerv_stiffness(inp$pm, out$qm, out$rev_kl, lambda = method$cost$lambda,
                     beta = method$kernel$beta, eps = method$eps)
    },
    out_updated_fn = klqp_update
  )
  method <- on_inp_updated(method, nerv_inp_update)$method
  method
}

# NeRV with uniform bandwidth (UNeRV)
#
# This method behaves like \code{nerv} in terms of its cost function,
# but treats the output weighting function like \code{asne} and
# \code{ssne} by setting the weight function decay parameter
# \code{beta} to 1. If you want to compare the NeRV cost function with the ASNE
# cost function directly, UNeRV is a better method to use than NeRV.
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to ASNE. Must be a value between 0 and 1.
# @param beta Controls the rate of decay of the exponential similarity kernel
#  function. Leave at the default value of 1 to compare with SSNE and ASNE.
# @param eps Small floating point value used to prevent numerical problems,
#   e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{nerv} uses the precisions that generated the input
# probability as a way to reflect the differences in the local density of
# points in the input space in the output embedding.
# The return value of this function should be used with the
# \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
unerv <- function(lambda = 0.5, beta = 1, eps = .Machine$double.eps,
                  verbose = TRUE) {
  lreplace(
    asne(beta = beta, eps = eps, verbose = verbose),
    cost = nerv_fg(lambda = lambda),
    stiffness_fn = function(method, inp, out) {
      nerv_stiffness(inp$pm, out$qm, out$rev_kl, lambda = method$cost$lambda,
                     beta = method$kernel$beta, eps = method$eps)
    },
    out_updated_fn = klqp_update
  )
}

# Symmetric Neighbor Retrieval Visualizer (SNeRV)
#
# A probability-based embedding method.
#
# SNeRV is a "symmetric" variant of \code{nerv}. Rather than use the
# conditional point-based probabilities of \code{asne}, it uses the
# joint pair-based probabilities of \code{ssne}. However, it uses a
# non-uniform exponential kernel in its output space (the decay
# parameter \code{beta} is allowed to vary per point by using the value
# calculated from the input data). As a result, there is an extra step required
# to produce the joint probability output: like the input probability matrix,
# the joint probabilities are generated from the conditional probabilities by
# averaging \code{q[i, j]} and \code{q[j, i]}.
#
# The probability matrix used in SNeRV:
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
#  \item{\code{qm}}{Joint probability matrix.}
# }
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
# @param eps Small floating point value used to prevent numerical problems,
#   e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
# Information retrieval perspective to nonlinear dimensionality reduction for
# data visualization.
#
# @seealso SNeRV uses the \code{nerv_cost} cost function and the
#   \code{exp_weight} similarity function for converting distances to
#   probabilities.
# The return value of this function should be used with the
#   \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# # default SNeRV settings
# embed_prob(method = usnerv(lambda = 0.5), ...)
# }
snerv <- function(lambda = 0.5, eps = .Machine$double.eps, verbose = TRUE) {
  snerv_plugin(lambda = lambda, eps = eps)
}

# Symmetric Neighbor Retrieval Visualizer with uniform bandwidths (USNeRV)
#
# A probability-based embedding method.
#
# USNeRV is a "symmetric" variant of \code{unerv}. Rather than use the
# conditional point-based probabilities of \code{asne}, it uses the
# joint pair-based probabilities of \code{ssne}. It differs from
# \code{snerv} by only using a uniform kernel decay parameter to generate
# the output weight matrix.
#
# When \code{lambda = 1}, this method is equivalent to SSNE.
#
# The probability matrix used in SNeRV:
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
#  \item{\code{qm}}{Joint probability matrix.}
# }
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
# @param beta Decay parameter of the exponential similarity kernel
#  function. The larger the value, the faster the function decreases.
# @param eps Small floating point value used to prevent numerical problems,
#   e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @seealso \code{snerv} is a version of this algorithm which relaxes
# the requirement for uniform precisions in the output weight kernel (at
# the cost of being somewhat less efficient).
#
# The return value of this function should be used with the
#   \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# # default SNeRV settings
# embed_prob(method = usnerv(lambda = 0.5), ...)
#
# # equivalent to SSNE
# embed_prob(method = usnerv(lambda = 1), ...)
#
# # puts an emphasis on only keeping true neighbors close together
# # tends to produce a larger number of small, tight clusters
# embed_prob(method = usnerv(lambda = 0), ...)
# }
usnerv <- function(lambda = 0.5, beta = 1, eps = .Machine$double.eps,
                   verbose = TRUE) {
  lreplace(
    unerv(lambda = lambda, beta = beta, eps = eps, verbose = verbose),
    stiffness_fn = function(method, inp, out) {
      usnerv_stiffness(inp$pm, out$qm, out$rev_kl, lambda = method$cost$lambda,
                      beta = method$kernel$beta, eps = method$eps)
    },
    prob_type = "joint"
  )
}

# Heavy-tailed Symmetric Neighbor Retrieval Visualizer (HSNeRV)
#
# A probability-based embedding method.
#
# HSNeRV is a hybrid of \code{snerv} and \code{hssne}. It has
# the \code{lambda} parameter of SNeRV, allowing for the control of precision
# versus recall, and the \code{alpha} parameter of HSSNE which
# give the behavior of SNeRV when \code{alpha} is close to zero,
# and behavior somewhat like that of t-NeRV when \code{alpha = 1}.
#
# Like NeRV and SNeRV, the kernel parameters are non-uniform and taken from the
# input data.
#
# The probability matrix used in HSNeRV:
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
# }
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
# @param alpha Tail heaviness. Must be greater than zero. When set to a small
#   value this method is equivalent to SSNE or SNeRV (depending on the value
#   of \code{lambda}. When set to one to one, this method behaves like
#   t-SNE/t-NeRV.
# @param eps Small floating point value used to prevent numerical problems,
#   e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
# Information retrieval perspective to nonlinear dimensionality reduction for
# data visualization.
# \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#
# Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
# Heavy-tailed symmetric stochastic neighbor embedding.
# In \emph{Advances in neural information processing systems} (pp. 2169-2177).
# @seealso HSNeRV uses the \code{nerv_cost} cost function and the
#   \code{heavy_tail_weight} similarity function for converting
#   distances to probabilities.
#
# The return value of this function should be used with the
#   \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# # equivalent to default SNeRV
# embed_prob(method = hsnerv(lambda = 0.5, alpha = 1.5e-8), ...)
#
# }
hsnerv <- function(lambda = 0.5, alpha = 0,
                   eps = .Machine$double.eps, verbose = TRUE) {
  hsnerv_plugin(lambda = lambda, alpha = alpha, eps = eps)
}

# Heavy-tailed SNeRV with Uniform Kernel Parameters (UHSNeRV)
#
# A probability-based embedding method.
#
# UHSNeRV is a hybrid of \code{usnerv} and \code{hssne}. It has
# the \code{lambda} parameter of USNeRV, allowing for the control of precision
# versus recall, and the \code{alpha} and \code{beta} parameters of HSSNE which
# give the behavior of SSNE/SNeRV when \code{alpha} is close to zero,
# and of t-SNE/t-NeRV when \code{alpha = 1}.
#
# The probability matrix used in HSNeRV:
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
# }
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
# @param alpha Tail heaviness. Must be greater than zero. When set to a small
#   value this method is equivalent to SSNE or SNeRV (depending on the value
#   of \code{lambda}. When set to one to one, this method behaves like
#   t-SNE/t-NeRV.
# @param beta Controls the rate of decay of the function. Equivalent to the
#   exponential decay parameter when \code{alpha} approaches 0.
# @param eps Small floating point value used to prevent numerical problems,
#   e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
# Information retrieval perspective to nonlinear dimensionality reduction for
# data visualization.
# \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#
# Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
# Heavy-tailed symmetric stochastic neighbor embedding.
# In \emph{Advances in neural information processing systems} (pp. 2169-2177).
# @seealso HSNeRV uses the \code{nerv_cost} cost function and the
#   \code{heavy_tail_weight} similarity function for converting
#   distances to probabilities.
#
# The return value of this function should be used with the
#   \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# # equivalent to SSNE
# embed_prob(method = hsnerv(lambda = 1, alpha = 1.5e-8), ...)
#
# # equivalent to t-SNE
# embed_prob(method = hsnerv(lambda = 1, alpha = 1), ...)
#
# # equivalent to default SNeRV
# embed_prob(method = hsnerv(lambda = 0.5, alpha = 1.5e-8), ...)
#
# # equivalent to default t-NeRV
# embed_prob(method = hsnerv(lambda = 0.5, alpha = 1), ...)
#
# }
uhsnerv <- function(lambda = 0.5, alpha = 0, beta = 1,
                    eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    usnerv(lambda = lambda, eps = eps, verbose = verbose),
    kernel = heavy_tail_kernel(beta = beta, alpha = alpha),
    stiffness_fn = function(method, inp, out) {
      uhsnerv_stiffness(pm = inp$pm, qm = out$qm, wm = out$wm,
                       rev_kl = out$rev_kl,
                       lambda = method$cost$lambda, alpha = method$kernel$alpha,
                       beta = method$kernel$beta, eps = method$eps)
    },
    update_out_fn = make_update_out(keep = c("qm", "wm"))
  )
}

# t-Distributed Neighbor Retrieval Visualizer (t-NeRV)
#
# A probability-based embedding method.
#
# t-NeRV is a variant of t-distributed Stochastic Neighbor Embedding
# (\code{tsne}), with a modified cost function: in addition to
# calculating the Kullback-Leibler divergence of the output probabilities Q,
# from the input probabilities, P, it also includes the divergence of P from Q.
# The final cost function is a weighted sum of these two individual functions.
# Hence SSNE is a special case of NeRV where all the weight is placed on the
# first component of the cost function.
#
# From an information retrieval perspective, the weighting factor allows the
# user to place a relative weight on false positives: points on the embedded
# map which have a close distance, but a low input probability, i.e. should not
# have been embedded as close neighbors, versus false negatives: pairs with a
# large distance in the output coordinates, but a high input probability, i.e.
# should have been embedded as close neighbors. From this perspective, t-SNE
# is the equivalent of emphasising false positives over false negatives.
#
# The probability matrix used in t-NeRV:
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
# }
# @param lambda Weighting factor controlling the emphasis placed on precision
#   (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#   1, then the method is equivalent to t-SNE. Must be a value between 0 and 1.
# @param eps Small floating point value used to prevent numerical problems,
#   e.g. in gradients and cost functions.
# @param verbose If \code{TRUE}, log information about the embedding.
# @return An embedding method for use by an embedding function.
# @references
# Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
# Information retrieval perspective to nonlinear dimensionality reduction for
# data visualization.
# \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
# @seealso NeRV uses the \code{nerv_cost} cost function and the
#   \code{tdist_weight} similarity function for converting distances to
#   probabilities.
# The return value of this function should be used with the
#  \code{embed_prob} embedding function.
# @family sneer embedding methods
# @family sneer probability embedding methods
# @examples
# \dontrun{
# # default t-NeRV settings
# embed_prob(method = tnerv(lambda = 0.5), ...)
#
# # equivalent to t-SNE
# embed_prob(method = tnerv(lambda = 1), ...)
#
# # puts an emphasis on precision over recall and allows long tails
# # will create widely-separated small clusters
# embed_prob(method = tnerv(lambda = 0), ...)
# }
tnerv <- function(lambda = 0.5, eps = .Machine$double.eps, verbose = TRUE) {
  lreplace(
    tsne(eps = eps, verbose = verbose),
    cost = nerv_fg(lambda = lambda),
    stiffness_fn = function(method, inp, out) {
      tnerv_stiffness(inp$pm, out$qm, out$wm, out$rev_kl,
                      lambda = method$cost$lambda, eps = method$eps)
    },
    out_updated_fn = klqp_update
  )
}

# NeRV Stiffness Function
#
# @param pm Input probability matrix.
# @param qm Output probabilty matrix.
# @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
# @param lambda NeRV weighting factor controlling the emphasis placed on
# precision versus recall.
# @param beta Decay parameter of the weighting function.
# @param eps Small floating point value used to avoid numerical problems.
# @return Stiffness matrix.
nerv_stiffness <- function(pm, qm, rev_kl, lambda = 0.5, beta = 1,
                           eps = .Machine$double.eps) {
  (lambda * asne_stiffness(pm, qm, beta = beta)) +
    ((1 - lambda) * reverse_asne_stiffness(pm, qm, rev_kl, beta = beta,
                                           eps = eps))
}

# USNeRV Stiffness Function
#
# If using uniform decay parameters, the stiffness function for USNeRV is
# simplified compared to the generic non-uniform case for SNeRV.
#
# @param pm Input joint probability matrix.
# @param qm Output joint probabilty matrix.
# @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
# @param beta Decay parameter of the exponential weighting function.
# @param lambda NeRV weighting factor controlling the emphasis placed on
# precision versus recall.
# @param eps Small floating point value used to avoid numerical problems.
# @return Stiffness matrix
usnerv_stiffness <- function(pm, qm, rev_kl, beta = 1, lambda = 0.5,
                            eps = .Machine$double.eps) {
  (lambda * ssne_stiffness(pm, qm, beta = beta)) +
    ((1 - lambda) * reverse_ssne_stiffness(pm, qm, rev_kl, beta = beta,
                                           eps = eps))
}

# t-NeRV Stiffness Function
#
# @param pm Input joint probability matrix.
# @param qm Output joint probabilty matrix.
# @param wm Output weight probability matrix.
# @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
# @param lambda NeRV weighting factor controlling the emphasis placed on
# precision versus recall.
# @param eps Small floating point value used to avoid numerical problems.
# @return Stiffness matrix
tnerv_stiffness <- function(pm, qm, wm, rev_kl, lambda = 0.5,
                            eps = .Machine$double.eps) {
  (lambda * tsne_stiffness(pm, qm, wm)) +
    ((1 - lambda) * reverse_tsne_stiffness(pm, qm, wm, rev_kl, eps = eps))
}

# UHSNeRV Stiffness Function
#
# If using uniform decay parameters, the stiffness function for UHSNeRV is
# simplified compared to the generic non-uniform case for HSNeRV.
#
# @param pm Input joint probability matrix.
# @param qm Output joint probabilty matrix.
# @param wm Output weight probability matrix.
# @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
# @param lambda NeRV weighting factor controlling the emphasis placed on
# precision versus recall.
# @param alpha Tail heaviness of the weighting function.
# @param beta The decay parameter of the weighting function, equivalent to
#  the exponential decay parameter when \code{alpha} approaches zero.
# @param eps Small floating point value used to avoid numerical problems.
# @return Stiffness matrix.
uhsnerv_stiffness <- function(pm, qm, wm, rev_kl, lambda = 0.5,
                             alpha = 1.5e-8, beta = 1,
                             eps = .Machine$double.eps) {
  (lambda * hssne_stiffness(pm, qm, wm, alpha = alpha, beta = beta)) +
    ((1 - lambda) * reverse_hssne_stiffness(pm, qm, wm, rev_kl,
                                            alpha = alpha, beta = beta,
                                            eps = eps))
}

# Neighbor Retrieval Visualizer (NeRV) Cost Function
#
# A measure of embedding quality between input and output data.
#
# This cost function evaluates the embedding quality by calculating a weighted
# sum of two KL divergence calculations:
#
# \deqn{C_{NeRV} = \lambda D_{KL}(P||Q) + (1-\lambda)D_{KL}(Q||P)}{C_NeRV = [lambda * KL(P||Q)] + [(1-lambda) * KL(Q||P)]}
#
# where P is the input probability matrix, Q the output probability matrix, and
# \eqn{\lambda}{lambda} is a weighting factor between zero and one.
#
# This cost function requires the following matrices and values to be defined:
# \describe{
#  \item{\code{inp$pm}}{Input probabilities.}
#  \item{\code{out$qm}}{Output probabilities.}
#  \item{\code{method$cost$lambda}}{Weighting factor between 0 and 1.}
# }
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return NeRV cost.
# @references
# Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
# Information retrieval perspective to nonlinear dimensionality reduction for
# data visualization.
# \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
# @family sneer cost functions
nerv_cost <- function(inp, out, method) {
  method$cost$lambda * kl_cost(inp, out, method) +
    (1 - method$cost$lambda) * reverse_kl_cost(inp, out, method)
}
attr(nerv_cost, "sneer_cost_type") <- "prob"


# Reverse Kullback-Leibler Divergence Cost Function
#
# A measure of embedding quality between input and output data.
#
# This cost function the embedding quality by calculating the KL divergence
# between the input probabilities and the output probabilities, where the
# output probabilities are considered the reference probabilities.
#
# This cost function requires the following matrices to be defined:
# \describe{
#  \item{\code{inp$pm}}{Input probabilities.}
#  \item{\code{out$qm}}{Output probabilities.}
# }
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return KL divergence between \code{inp$pm} and \code{out$qm}.
# @seealso \code{kl_cost} provides more detail on the differences
#   between the usual KL divergence and this "reverse" divergence.
# @family sneer cost functions
reverse_kl_cost <- function(inp, out, method) {
  kl_divergence(out$qm, inp$pm, method$eps)
}
attr(reverse_kl_cost, "sneer_cost_type") <- "prob"

# Reverse Kullback Leibler Divergence Cost
#
# Cost wrapper factory function.
#
# Creates the a list containing the required functions for using "reverse"
# Kullback Leibler Divergence, i.e. KL(Q||P), in an embedding.
#
# Provides the cost function and its gradient (with respect to Q).
#
# @return KL divergence function and gradient.
# @family sneer cost wrappers
reverse_kl_fg <- function() {
  list(
    fn = reverse_kl_cost,
    gr = reverse_kl_cost_gr,
    name = "Reverse KL"
  )
}

# Reverse Kullback Leibler Divergence Cost Gradient
#
# Calculates the gradient of the "reverse" KL divergence cost function of
# an embedding with respect to the output probabilities.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Gradient of the reverse KL divergence cost of the embedding with
# respect to the output probabilities.
reverse_kl_cost_gr <- function(inp, out, method) {
  reverse_kl_divergence_gr(out$qm, inp$pm, method$eps)
}

# Reverse Kullback Leibler Gradient
#
# Calculates the gradient of the KL divergence with respect to the
# probability Q in KL(Q||P).
#
# @param pm Probability Matrix. First probability in the divergence.
# @param qm Probability Matrix. Second probability in the divergence.
# @param eps Small floating point value used to avoid numerical problems.
# @return Gradient of the KL divergence from \code{qm} to \code{pm}.
reverse_kl_divergence_gr <- function(pm, qm, eps = .Machine$double.eps) {
  log((pm + eps) / (qm + eps)) + 1
}


# Set Output Kernel Parameter From Input Results
#
# Updates the output kernel in response to a change in input probability.
#
# This function is called when the input probability has changed. It transfers
# the precision parameters from the input data to the output kernel. This is
# used in the NeRV family of embedding routines where the precisions of the
# output exponential kernel are set to those of the input kernel.
#
# This function expects:
# \itemize{
#  \item{The \code{inp} list contains a member called \code{beta}.}
#  \item{\code{beta} is a vector of numeric parameters with the same length
#  as the size of output squared distance matrix.}
#  \item{The \code{out$kernel} has a \code{beta} parameter which can make
#  use of a vector of parameters.}
# }
#
# These conditions are all satisifed if you use an exponential kernel for
# creating the input data, and an asymmetric exponential kernel for the output
# data, as in the usual NeRV functions. If you deviate from these conditions,
# you may get incorrect behavior.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Updated kernel.
# @family sneer kernel modifiers
transfer_kernel_precisions <- function(inp, out, method) {
  if (method$verbose) {
    message("Transferring input precisions to output kernel")
  }
  method$kernel$beta <- inp[["beta"]]
  method$kernel <- check_symmetry(method$kernel)
  method$kernel
}

# NeRV input update function
#
# Update function to run when the input data has changed. Unlike most other
# probability-based embedding methods, NeRV sets the output kernels to have
# the same parameters as the equivalent input kernels (in the case of NeRV, the
# exponential parameter).
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return List containing the update embedding method, where the kernel beta
# parameters have been updated to be the same as that for the \code{input}
# data.
nerv_inp_update <- function(inp, out, method) {
  method$kernel <- transfer_kernel_precisions(inp, out, method)
  list(method = method)
}

# Updates the Kullback Leibler Divergence.
#
# Calculates and stores the KL divergence from P (input probabilities) to Q
# (output probabilities) on the output data. Used by those embedding methods
# where the KL divergence is used to calculate the stiffness matrix in a
# gradient calculation (e.g. \code{nerv}).
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return \code{out} updated with the KL divergence from {\code{inp$pm}} to
# \code{out$qm}.
klqp_update <- function(inp, out, method) {
  prob_type <- method$prob_type
  if (is.null(prob_type)) {
    stop("Embedding method must have a prob type")
  }
  fn_name <- paste0("klqp_update_p", prob_type)
  fn <- get(fn_name)
  if (is.null(fn)) {
    stop("Unable to find KLQP update function for ", prob_type)
  }
  fn(inp, out, method)
}

# Updates the Kullback Leibler Divergence for Joint Probabilities.
#
# Calculates and stores the KL divergence from P (input probabilities) to Q
# (output probabilities) on the output data. Used by those embedding methods
# where the KL divergence is used to calculate the stiffness matrix in a
# gradient calculation (e.g. \code{snerv}).
#
# Only appropriate for embedding methods that use joint probabilities.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return \code{out} updated with the KL divergence from {\code{inp$pm}} to
# \code{out$qm}.
klqp_update_pjoint <- function(inp, out, method) {
  out$rev_kl <- kl_divergence(out$qm, inp$pm, method$eps)
  out
}

# Updates the Kullback Leibler Divergence for Row Probabilities.
#
# Calculates and stores the KL divergence from P (input probabilities) to Q
# (output probabilities) on the output data. Used by those embedding methods
# where the KL divergence is used to calculate the stiffness matrix in a
# gradient calculation (e.g. \code{nerv}).
#
# Only appropriate for embedding methods that use row probabilities.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return \code{out} updated with the KL divergence from {\code{inp$pm}} to
# \code{out$qm}.
klqp_update_prow <- function(inp, out, method) {
  out$rev_kl <- kl_divergence_rows(out$qm, inp$pm, method$eps)
  out
}

# NeRV Cost
#
# Cost wrapper factory function.
#
# Creates the a list containing the required functions for using the NeRV cost
# in an embedding.
#
# Provides the cost function and its gradient (with respect to Q).
#
# @param lambda Weighting factor controlling the emphasis placed on precision
#  (set \code{lambda} to 0), versus recall (set \code{lambda} to 1). If set to
#  1, then the cost behaves like the Kullback Leibler divergence. If set to 0,
#  the cost behaves like the "reverse" KL divergence.
# @return NeRV cost function and gradient.
# @family sneer cost wrappers
nerv_fg <- function(lambda = 0.5) {
  fn <- function(inp, out, method) {
    nerv_cost(inp, out, method)
  }
  attr(fn, "sneer_cost_type") <- "prob"

  list(
    fn = fn,
    gr = nerv_cost_gr,
    lambda = lambda,
    name = "NeRV"
  )
}

# NeRV Cost Gradient
#
# Calculates the gradient of the NeRV cost of an embedding with respect to the
# output probabilities.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Gradient of the NeRV cost.
nerv_cost_gr <- function(inp, out, method) {
  method$cost$lambda * kl_cost_gr(inp, out, method) +
    (1 - method$cost$lambda) * reverse_kl_cost_gr(inp, out, method)
}
