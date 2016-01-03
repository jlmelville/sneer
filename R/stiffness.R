# Stiffness functions. Generally only valid for a specific cost function
# (and for probability-based embeddings, a specific cost function/weighting
# function pair). However, some stiffness functions can be written in terms of
# others.

#' ASNE Stiffness Function
#'
#' @param pm Input probability matrix.
#' @param qm Output probabilty matrix.
#' @return Stiffness matrix.
asne_stiffness <- function(pm, qm) {
  km <- pm - qm
  2 * (km + t(km))
}

#' SSNE Stiffness Function
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @return Stiffness matrix.
ssne_stiffness <- function(pm, qm) {
  4 * (pm - qm)
}

#' t-SNE Stiffness Function
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param wm Output weight probability matrix.
#' @return Stiffness matrix.
tsne_stiffness <- function(pm, qm, wm) {
  ssne_stiffness(pm, qm) * wm
}

#' t-ASNE Stiffness Function
#'
#' @param pm Input probability matrix.
#' @param qm Output probabilty matrix.
#' @param wm Output weight probability matrix.
#' @return Stiffness matrix.
tasne_stiffness <- function(pm, qm, wm) {
  km <- (pm - qm) * wm
  2 * (km + t(km))
}

#' HSSNE Stiffness Function
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param wm Output weight probability matrix.
#' @param alpha Tail heaviness of the weighting function.
#' @param beta The precision of the weighting function.
#' @return Stiffness matrix.
hssne_stiffness <- function(pm, qm, wm, alpha = 1.5e-8, beta = 1) {
  4 * beta * (pm - qm) * (wm ^ alpha)
}


#' "Reverse" ASNE Stiffness Function
#'
#' Uses the exponential weighting function for similarities, but the
#' "reverse" Kullback Leibler divergence as the cost function.
#'
#' @param pm Input probability matrix.
#' @param qm Output probabilty matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
rev_asne_stiffness <- function(pm, qm, rev_kl, eps = .Machine$double.eps) {
  km <- qm * (log(pm / (qm + eps)) + rev_kl)
  2 * (km + t(km))
}

#' NeRV Stiffness Function
#'
#' @param pm Input probability matrix.
#' @param qm Output probabilty matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param lambda NeRV weighting factor controlling the emphasis placed on
#' precision versus recall.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
nerv_stiffness <- function(pm, qm, rev_kl, lambda = 0.5,
                           eps = .Machine$double.eps) {
  (lambda * asne_stiffness(pm, qm)) +
    ((1 - lambda) * rev_asne_stiffness(pm, qm, rev_kl, eps))
}

#' "Reverse" SSNE Stiffness Function
#'
#' Uses the exponential weighting function for similarities, but the
#' "reverse" Kullback Leibler divergence as the cost function.
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
rev_ssne_stiffness <- function(pm, qm, rev_kl, eps = .Machine$double.eps) {
  4 * qm * (log(pm / (qm + eps)) + rev_kl)
}

#' SNeRV Stiffness Function
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param wm Output weight probability matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param lambda NeRV weighting factor controlling the emphasis placed on
#' precision versus recall.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix
snerv_stiffness <- function(pm, qm, wm, rev_kl, lambda = 0.5,
                            eps = .Machine$double.eps) {
  (lambda * ssne_stiffness(pm, qm)) +
    ((1 - lambda) * rev_ssne_stiffness(pm, qm, rev_kl, eps))
}

#' "Reverse" t-SNE Stiffness Function
#'
#' Uses the exponential weighting function for similarities, but the
#' "reverse" Kullback Leibler divergence as the cost function.
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param wm Output weight probability matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
rev_tsne_stiffness <- function(pm, qm, wm, rev_kl, eps = .Machine$double.eps) {
  rev_ssne_stiffness(pm, qm, rev_kl, eps) * wm
}

#' t-NeRV Stiffness Function
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param wm Output weight probability matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param lambda NeRV weighting factor controlling the emphasis placed on
#' precision versus recall.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix
tnerv_stiffness <- function(pm, qm, wm, rev_kl, lambda = 0.5,
                           eps = .Machine$double.eps) {
  (lambda * tsne_stiffness(pm, qm, wm)) +
    ((1 - lambda) * rev_tsne_stiffness(pm, qm, wm, rev_kl, eps))
}

#' "Reverse" HSSNE Stiffness Function
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param wm Output weight probability matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param alpha Tail heaviness of the weighting function.
#' @param beta The precision of the weighting function.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
rev_hssne_stiffness <- function(pm, qm, wm, rev_kl, alpha = 1.5e-8,
                                    beta = 1, eps = .Machine$double.eps) {
  4 * beta * qm * (log(pm / (qm + eps)) + rev_kl) * (wm ^ alpha)
}

#' HSNeRV Stiffness Function
#'
#' @param pm Input joint probability matrix.
#' @param qm Output joint probabilty matrix.
#' @param wm Output weight probability matrix.
#' @param rev_kl "Reverse" KL divergence between \code{pm} and \code{qm}.
#' @param lambda NeRV weighting factor controlling the emphasis placed on
#' precision versus recall.
#' @param alpha Tail heaviness of the weighting function.
#' @param beta The precision of the weighting function.
#' @param eps Small floating point value used to avoid numerical problems.
#' @return Stiffness matrix.
hsnerv_stiffness <- function(pm, qm, wm, rev_kl, lambda = 0.5,
                             alpha = 1.5e-8, beta = 1,
                             eps = .Machine$double.eps) {
  (lambda * hssne_stiffness(pm, qm, wm, alpha, beta)) +
    ((1 - lambda) * rev_hssne_stiffness(pm, qm, wm, rev_kl, alpha, beta, eps))
}
