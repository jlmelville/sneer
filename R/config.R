# Functions for creating embedding methods.
# Use make_embed_method for the most control.
# Methods like asne, tsne and ssne can be used with their default arguments.

#' Create an embedding method.
#'
#' Factory function for embedding methods. More usual to call specific embedding
#' method functions.
#'
#' @param prob_in_fn Function to calculate an input probability distribution.
#' Should take a distance matrix and return a probability matrix.
#' @param stiffness Stiffness configuration.
#' @return Embedding method: a list consisting of the passed in parameters.
make_embed_method <- function(prob_in_fn = NULL,
                              stiffness = tsne_stiffness()) {
  list(
    prob_in_fn = prob_in_fn,
    stiffness = stiffness
  )
}

#' Create the Asymmetric Stochastic Neighbor Embedding (ASNE) method.
#'
#' @return ASNE embedding method.
asne <- function(prob_in_fn = perp_prow,
                 stiffness = asne_stiffness()) {
  make_embed_method(prob_in_fn, stiffness)
}

#' Create the Symmetric Stochastic Neighbor Embedding (TSNE) method.
#'
#' @return SSNE embedding method.
ssne <- function(prob_in_fn = perp_pjoint,
                 stiffness = ssne_stiffness()) {
  make_embed_method(prob_in_fn, stiffness)
}

#' Create the T-Distributed Stochastic Neighbor Embedding (TSNE) method.
#'
#' @return TSNE embedding method.
tsne <- function(prob_in_fn = perp_pjoint,
                 stiffness = tsne_stiffness()) {
  make_embed_method(prob_in_fn, stiffness)
}
