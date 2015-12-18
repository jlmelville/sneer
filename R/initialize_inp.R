#' Create an input initialization function.
#'
#' @param d_to_p_fn Function to convert distances to probabilities. Should
#' have the signature: \code{d_to_p_fn(dm)} where \code{dm} is a distance
#' matrix. Should return a list containing \code{pm}: the probabilities.
#' @return Initialization function with signature \code{fn(xm)} where \code{xm}
#' is the input coordinates or a distance matrix. This function will return
#' a list containing: \code{xm} the input coordinates if these were provided as
#' input, \code{DX} input distances, \code{pm} the input probabilities,
#' if \code{d_to_p_fn} is non-null.
make_init_inp <- function(d_to_p_fn = NULL) {
  function(xm) {
    inp <- list()
    if (class(xm) == "dist") {
      inp$dxm <- clamp(as.matrix(xm))
    } else {
      inp$xm <- as.matrix(xm)
      inp$dxm <- distance_matrix(inp$xm)
    }

    if (!is.null(d_to_p_fn)) {
      inp$pm <- d_to_p_fn(inp$dxm)$pm
    }

    inp
  }
}


#' Create a function which creates a joint probability matrix from a
#' distance matrix.
#'
#' Used in SSNE and TSNE to create input probabilities.
#'
#' @param d_to_p_fn Function with signature \code{(DX)}, where \code{DX} is a
#' distance matrix, and returns a list containing at least: \code{pm}: a row
#' probability matrix.
#' @return List containing: \code{pm}: a joint probability matrix.
p_to_pjoint <- function(d_to_p_result) {
  d_to_p_result$pm <- pcond_to_pjoint(d_to_p_result$pm)
  d_to_p_result
}

#' Create a function which creates a conditional probability matrix from a
#' distance matrix.
#'
#' @param d_to_p_fn Function with signature \code{(DX)}, where \code{DX} is a
#' distance matrix, and returns a list containing at least: \code{pm}: a row-wise
#' probability matrix.
#' @return List containing: \code{pm}: a conditional probability matrix.
p_to_pcond <- function(d_to_p_result) {
  pm <- d_to_p_result$pm
  pm <- pm / sum(pm)
  d_to_p_result$pm <- clamp(pm)
  d_to_p_result
}

#' Creates a symmetric matrix from a square matrix.
#'
#' The matrix is symmetrized by setting \code{pm[i, j]} and \code{pm[j, i]} to
#' their average, i.e. \code{Pij} = \code{(Pij + Pji)/2} = \code{Pji}.
#'
#' In SSNE and TSNE, this is used as part of the process of converting the row
#' stochastic matrix of conditional input probabilities to a joint probability
#' matrix.
#'
#' @param pm Square matrix to symmetrize.
#' @return The symmetrized matrix such that \code{pm[i, j]} = \code{pm[j, i]}
symmetrize_matrix <- function(pm) {
  0.5 * (pm + t(pm))
}

#' Create a function which creates a row probability matrix from a
#' distance matrix.
#'
#' Used in SSNE and TSNE to create input probabilities.
#'
#' @param d_to_p_fn Function with signature \code{(dm)}, where \code{dm} is a
#' distance matrix, and returns a list containing at least: \code{pm}: a row
#' probability matrix.
#' @return List containing: \code{pm}: a row probability matrix.
p_to_prow <- function(d_to_p_result) {
  d_to_p_result$pm <- clamp(d_to_p_result$pm)
  d_to_p_result
}

perp_prow <- compose(p_to_prow, d_to_p_perp_bisect)

perp_pjoint <- compose(p_to_pjoint, d_to_p_perp_bisect)
