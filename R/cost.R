# Cost functions. Used by optimization routines to improve the embedding.

# Kullback-Leibler Divergence Cost Function
#
# A measure of embedding quality between input and output data.
#
# This cost function evaluates the embedding quality by calculating the KL
# divergence between the input probabilities and the output probabilities.
# More specifically, it considers the input probabilities to be the reference
# probabilities. See the note below for more details and whether you should
# care about the distinction.
#
# @note The KL divergence is asymmetric, so that D_KL(P||Q) != D_KL(Q||P).
# With this cost function, the input probability distribution is considered the
# "reference" probability, and a more precise way to describe this cost
# function is that it measures the divergence of the output probabilities
# \emph{from} from the input probabilities.
#
# For t-SNE and related embedding methods, only this type of KL divergence is
# calculated. However other methods (e.g. NeRV) also consider the "reverse"
# divergence, i.e. using the output probabilities as reference probabilities.
# Equivalently, this could be defined as the KL divergence of the input
# probabilities \emph{from} the output probabilities.
#
# This cost function requires the following matrices to be defined:
# \describe{
#  \item{\code{inp$pm}}{Input probabilities.}
#  \item{\code{out$qm}}{Output probabilities.}
# }
#
# For embedding methods which define their cost function over multiple
# probability distributions (e.g. \code{\link{asne}}), this cost function
# returns the sum of the divergences.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return KL divergence between \code{inp$pm} and \code{out$qm}.
# @seealso To use \code{out$qm} as the reference probability and calculate the
#   divergence of \code{inp$pm} from \code{out$qm}, see
#   \code{\link{reverse_kl_cost}}.
# @family sneer cost functions
kl_cost <- function(inp, out, method) {
  kl_divergence(inp$pm, out$qm, method$eps)
}
attr(kl_cost, "sneer_cost_type") <- "prob"


# Kullback-Leibler Divergence
#
# A measure of embedding quality between input and output probability matrices.
#
# The Kullback-Leibler Divergence between two discrete probabilities P and Q
# is:
#
# \deqn{D_{KL}(P||Q) = \sum_{i}P(i)\log\frac{P(i)}{Q(i)}}{D_KL(P||Q) = sum(Pi*log(Pi/Qi))}
#
# The base of the log determines the units of the divergence.
#
# If a row probability matrix is provided (where each row in the matrix is
# a separate distribution), this function returns the sum of all divergences.
#
# @param pm Probability Matrix. First probability in the divergence.
# @param qm Probability Matrix. Second probability in the divergence.
# @param eps Small floating point value used to avoid numerical problems.
# @return KL divergence between \code{pm} and \code{qm}.
# @seealso \code{\link{kl_divergence_rows}} to obtain the separate KL
# divergences when using row probability matrices.
kl_divergence <- function(pm, qm, eps = .Machine$double.eps) {
  sum(kl_divergence_rows(pm, qm, eps))
}

# Kullback-Leibler Divergence per Row
#
# A measure of embedding quality between input and output row probability
# matrices.
#
# The Kullback-Leibler Divergence between two discrete probabilities P and Q
# is:
#
# \deqn{D_{KL}(P||Q) = \sum_{i}P(i)\log\frac{P(i)}{Q(i)}}{D_KL(P||Q) = sum(Pi*log(Pi/Qi))}
#
# The base of the log determines the units of the divergence.
#
# This function calculates the KL for each distribution in the provided
# matrices, one per row.
#
# @param pm Row probability Matrix. First probability in the divergence.
# @param qm Row Probability Matrix. Second probability in the divergence.
# @param eps Small floating point value used to avoid numerical problems.
# @return Vector of KL divergences from \code{qm} to \code{pm}.
kl_divergence_rows <- function(pm, qm, eps = .Machine$double.eps) {
  apply(pm * log((pm + eps) / (qm + eps)), 1, sum)
}

# Create Cost Function Normalizer
#
# A function to transform a cost function into a normalized cost function.
#
# The cost function can be any function where the more positive the value,
# the worse the solution is considered to be. The corresponding normalized
# version is that which scales the cost so that a "null" model would give a
# normalized cost of 1.0.
#
# The definition of a "null" model is one which is as good as can be if one
# didn't use any information from the data at all. For methods that attempt
# to preserve distances, this would be equivalent to making all the embedded
# distances the same, which can only be achieved by making them all zero. For
# probability-based methods, the equivalent would be to make all the
# probabilities equal.
#
# The cost function should have the signature \code{cost_fn(inp, out, method)}
# and return a scalar numeric cost value. In addition it should have an
# appropriate \code{sneer_cost_type} attribute set. For cost functions that act
# on probabilities, this should be \code{"prob"}. For cost function that act on
# distances, this should be \code{"dist"}.
#
# Note that this function will attempt to synthesize a function to calculate
# a suitable normalization value, but isn't very sophisticated: it simply sets
# the output probabilities or distances (depending on the type of the cost)
# function to a uniform value. If this is insufficiently clever, you can set
# the a \code{sneer_cost_norm} attribute instead. The value of this attribute
# should be the name of a separate cost function which does the normalization
# explicitly. In this case, the function won't attempt to synthesize a
# function, but will return that custom function instead.
#
# @param cost_fn Cost function to normalize.
# @return Normalized cost function with the signature
#  \code{norm_fn(inp, out, method)} and which returns a scalar numeric cost
#  value.
make_normalized_cost_fn <- function(cost_fn) {
  # see if there's a normalized cost function already defined
  norm_fn_name <- attr(cost_fn, "sneer_cost_norm")
  if (!is.null(norm_fn_name)) {
    norm_fn <- get(norm_fn_name)
    if (is.null(norm_fn)) {
      stop("No normalized cost function: ", norm_fn_name, " could be found")
    }
    return(norm_fn)
  }

  # otherwise, synthesize from the cost type
  cost_type <- attr(cost_fn, "sneer_cost_type")
  if (is.null(cost_type)) {
    stop("Cost function has no sneer_cost_type")
  }

  null_model_fn_name <- paste("null_model", cost_type, sep = "_")
  if (cost_type == "prob") {
    mat_name <- "qm"
  }
  else if (cost_type == "dist") {
    mat_name <- "dm"
  }
  else if (cost_type == "weight") {
    mat_name <- "wm"
  }
  else {
    stop("No known null model matrix name for cost type '", cost_type, "'")
  }

  function(inp, out, method) {
    cost <- cost_fn(inp, out, method)
    out[[mat_name]] <- do.call(null_model_fn_name, list(out[[mat_name]]))
    null_cost <- cost_fn(inp, out, method)
    cost / null_cost
  }
}

# Null Model for Probability Matrices
#
# For a given probability matrix, return the equivalent "null" model, i.e. one
# where all probabilities are equal.
#
# @param pm Probability matrix. Can be row, joint or conditional.
# @return Probability matrix where all elements are equal.
null_model_prob <- function(pm) {
  matrix(sum(pm) / (nrow(pm) * ncol(pm)), nrow = nrow(pm), ncol = ncol(pm))
}

# Kullback-Leibler Divergence Cost Wrapper Factory Function
#
# Cost wrapper factory function.
#
# Creates the a list containing the required functions for using the Kullback
# Leibler divergence, KL(P||Q), in an embedding.
#
# Provides the cost function and its gradient (with respect to Q).
#
# @return KL divergence function and gradient.
# @family sneer cost wrappers
kl_fg <- function() {
  list(
    fn = kl_cost,
    gr = kl_cost_gr,
    name = "KL"
  )
}

# Kullback Leibler Cost Gradients
#
# Measures the gradient of the KL divergence of an embedding, with respect
# to the probabilities of the ouput probabilities.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Gradient of the KL divergence from \code{inp$pm} to \code{out$qm}.
kl_cost_gr <- function(inp, out, method) {
  kl_divergence_gr(inp$pm, out$qm, method$eps)
}

# Kullback Leibler Gradient
#
# Calculates the gradient of the KL divergence with respect to the
# probability Q in KL(P||Q).
#
# @param pm Probability Matrix. First probability in the divergence.
# @param qm Probability Matrix. Second probability in the divergence.
# @param eps Small floating point value used to avoid numerical problems.
# @return Gradient of the KL divergence from \code{pm} to \code{qm}.
kl_divergence_gr <- function(pm, qm, eps = .Machine$double.eps) {
  -pm / (qm + eps)
}

# Finite Difference Gradient Calculation
#
# Calculate the gradient of the cost function for a specified position using
# a finite difference.
#
# Only intended for testing that analytical gradients have been calculated
# correctly.
#
# @param inp Input data.
# @param out Output data containing the desired position.
# @param method Embedding method.
# @param diff Step size to take in finite difference calculation.
# @return Gradient matrix.
cost_gradient_fd <- function(inp, out, method, diff = 1e-4) {
  qm <- out$qm
  nr <- nrow(qm)
  nc <- ncol(qm)

  grad <- matrix(0, nrow = nr, ncol = nc)
  for (i in 1:nr) {
    for (j in 1:nc) {
      old <- qm[i, j]
      qm[i, j] <- old - diff
      out$qm <- qm
      if (!is.null(method$out_updated_fn)) {
        out <- method$out_updated_fn(inp, out, method)
      }
      cost_back <- calculate_cost(method, inp, out)

      qm[i, j] <- old + diff
      out$qm <- qm
      if (!is.null(method$out_updated_fn)) {
        out <- method$out_updated_fn(inp, out, method)
      }
      cost_fwd <- calculate_cost(method, inp, out)

      fd <- (cost_fwd - cost_back) / (2 * diff)
      grad[i, j] <- fd
    }
  }
  grad
}

# Evaluate Cost Function
#
# @param method Embedding method containing a cost function.
# @param inp Input data.
# @param out Output data.
# @return Scalar cost function value.
calculate_cost <- function(method, inp, out) {
  method$cost$fn(inp, out, method)
}
