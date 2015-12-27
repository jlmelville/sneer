# Distance-preserving embedding methods like metric MDS and Sammon Mapping.

#' Metric Multi-dimensional scaling (MDS) using the STRESS cost function.
#'
#' Creates a list of functions that collectively implement metric MDS.
#'
#' This function minimizes an embedding using \code{\link{metric_stress_cost}}
#' as the cost function:
#'
#' \deqn{STRESS = \sum_{i<j} (r_{ij} - d_{ij})^2}{STRESS = sum(rij-dij)^2}
#'
#' \eqn{r_{ij}}{rij} is the input distance between point \eqn{i} and point
#' \eqn{j} and \eqn{d_{ij}}{dij} is the corresponding output distance.
#'
#' MDS is an umbrella term for many algorithms, with an emphasis on
#' non-metric problems, i.e. those where the input distance is not a metric.
#' The non-metric stress loss function has a slightly different definition, so
#' when comparing the result of this embedding method with that of other MDS
#' algorithms, you need to be quite sure exactly how the stress is being
#' calculated, which may not be apparent without examining the source code.
#'
#' @seealso
#' There are a plethora of MDS algorithms implemented in R. For non-metric MDS,
#' the most commonly reported loss function is the Kruskal
#' Stress (also known as Stress-1), which when applied to a metric problem is:
#'
#' \deqn{K = \sqrt{\frac{\sum_{i<j} (r_{ij} - d_{ij})^2}{\sum_{i<j} d_{ij}^2}}}{K = sqrt(sum(rij-dij)^2/sum(dij^2))}
#'
#' i.e. the square root of a normalized metric stress, where the normalization
#' is by the sum of the squares of the embedded distances. Some functions in other
#' packages which use this include:
#' \describe{
#' \item{\code{\link[MASS]{isoMDS}}}{Also applies a monotonic
#' transformation to the input distances to lower the stress, so the reported
#' stress and the output configuration can't be directly compared to the output
#' of a sneer MDS embedding.}
#' \item{\code{\link[smacof]{mds}}}{Applies ratio metric MDS, which attempts
#' to preserve the ratio of the distances in the input and output space. The
#' result of this function is a list, containin a member \code{stress}, which
#' is the Kruskal Stress, so can be compared to the
#' \code{\link{kruskal_stress_cost}} of a sneer embedding. However, the embedded
#' configuration in the result, \code{conf} configuration, is will not on the
#' same scale as the input coordinates.}
#' \item{\code{\link[stats]{cmdscale}}}{Doesn't use Kruskal Stress, but
#' implements a form of classical metric MDS called Principal Coordinate
#' Analysis. However, note that if the input distances are Euclidean (which
#' they are in sneer), the result is identical to PCA, so the output will
#' always have a higher stress than the sneer MDS embedding.}
#' }
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates. This name can be changed by
#'  specifying \code{mat_name}.}
#'  \item{\code{dm}}{Distance matrix generated from \code{ym}.}
#' }
#'
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return a list containing:
#'  \item{\code{cost_fn}}{Cost function for the embedding: for this
#'  implementation of MDS, the unweighted residual sum of squares between
#'  the distances.}
#'  \item{\code{stiffness_fn}}{Stiffness function for \code{cost_fn}.}
#'  \item{\code{update_out_fn}}{Function to calculate and store any needed
#'  data after a coordinate update.}
#'  \item{\code{mat_name}}{Name of the matrix in the output data list that will
#'  contain the embedded coordinates.}
#'  \item{\code{eps}}{Small floating point value used to prevent numerical
#'  problems, e.g. in gradients and cost functions.}
#' @family sneer embedding methods
#' @export
mmds <- function(mat_name = "ym", eps = .Machine$double.eps) {
  f <- function(dxm, dym, eps = .Machine$double.eps) {
    dym <- dym + eps
    -2 * (dxm - dym) / (dym + eps)
  }

  list(
    cost_fn = metric_stress_cost,
    stiffness_fn = function(method, inp, out) {
      f(inp$dm, out$dm, eps = method$eps)
    },
    update_out_fn = function(inp, out, method) {
      out$dm <- distance_matrix(out[[method$mat_name]])
      out
    },
    mat_name = mat_name,
    eps = eps
  )
}

#' Metric Multi-dimensional scaling (MDS) using SSTRESS.
#'
#' Embedding method.
#'
#' This function minimizes an embedding using the SSTRESS loss function:
#'
#' \deqn{SSTRESS = \sum_{i<j} ((r_{ij}^2 - d_{ij})^2)^2}{SSTRESS = sum(rij^2-dij^2)^2}
#'
#' \eqn{r_{ij}}{rij} is the input distance between point \eqn{i} and point
#' \eqn{j} and \eqn{d_{ij}}{dij} is the corresponding output distance.
#'
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return a list containing:
#'  \item{\code{cost_fn}}{Cost function for the embedding: for this
#'  implementation of MDS, the SSTRESS cost function.}
#'  \item{\code{stiffness_fn}}{Stiffness function for \code{cost_fn}.}
#'  \item{\code{update_out_fn}}{Function to calculate and store any needed
#'  data after a coordinate update.}
#'  \item{\code{mat_name}}{Name of the matrix in the output data list that will
#'  contain the embedded coordinates.}
#'  \item{\code{eps}}{Small floating point value used to prevent numerical
#'  problems, e.g. in gradients and cost functions.}
#' @family sneer embedding methods
#' @export
smmds <- function(mat_name = "ym", eps = .Machine$double.eps) {
  f <- function(dxm, dym, eps = .Machine$double.eps) {
    (-4 * dxm * (dxm ^ 2 - dym ^ 2)) / (dym + eps)
  }

  list(
    cost_fn = metric_sstress_cost,
    stiffness_fn = function(method, inp, out) {
      f(inp$dm, out$dm, eps = method$eps)
    },
    update_out_fn = function(inp, out, method) {
      out$dm <- distance_matrix(out[[method$mat_name]])
      out
    },
    mat_name = mat_name,
    eps = eps
  )
}

#' Sammon Mapping.
#'
#' Creates a list of functions that collectively implement Sammon Mapping.
#'
#' This function minimizes an embedding using a quadratic loss function on
#' the difference between the input distances and the output distances:
#'
#' \deqn{S = \frac{\sum_{i<j}\frac{(r_{ij} - d_{ij})^2}{r_{ij}}}
#' {\sum_{i<j} r_{ij}}}{S = sum(((rij-dij)/rij)^2)/sum(rij)}
#'
#' \eqn{r_{ij}}{rij} is the input distance between point \eqn{i} and point
#' \eqn{j} and \eqn{d_{ij}}{dij} is the corresponding output distance.
#'
#' This puts a greater emphasis on short distances over long distances, compared
#' to MDS. Note that the denominator is a constant, and is just there to
#' normalize the stress. It's not used for calculating the stiffness matrix
#' or the gradient, but is included in the calculation of the cost function.
#'
#' @section Output Data:
#' If used in an embedding, the output data list will contain:
#' \describe{
#'  \item{\code{ym}}{Embedded coordinates. This name can be changed by
#'  specifying \code{mat_name}.}
#'  \item{\code{dm}}{Distance matrix generated from \code{ym}.}
#' }
#'
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @param eps Small floating point value used to prevent numerical problems,
#' e.g. in gradients and cost functions.
#' @return a list containing:
#'  \item{\code{cost_fn}}{Cost function for the embedding: Sammon's stress.}
#'  \item{\code{stiffness_fn}}{Stiffness function for \code{cost_fn}. For
#'  Sammon's stress, we ignore the constant denominator.}
#'  \item{\code{update_out_fn}}{Function to calculate and store any needed
#'  data after a coordinate update.}
#'  \item{\code{mat_name}}{Name of the matrix in the output data list that will
#'  contain the embedded coordinates.}
#'  \item{\code{eps}}{Small floating point value used to prevent numerical
#'  problems, e.g. in gradients and cost functions.}
#' @seealso \code{\link[MASS]{sammon}}, which also carries out Sammon mapping.
#' Results should be comparable with those of a sneer embedding. The
#' \code{stress} value is equivalent to the \code{\link{sammon_stress_cost}}
#' function in sneer.
#' @family sneer embedding methods
#' @export
sammon_map <- function(mat_name = "ym", eps = .Machine$double.eps) {
  f <- function(dxm, dym, eps = .Machine$double.eps) {
    dxm <- dxm + eps
    dym <- dym + eps
    (-2 * (dxm - dym)) / (dym * dxm)
  }

  list(
    cost_fn = sammon_stress_cost,
    stiffness_fn = function(method, inp, out) {
      f(inp$dm, out$dm, eps = method$eps)
    },
    update_out_fn = function(inp, out, method) {
      out$dm <- distance_matrix(out[[method$mat_name]])
      out
    },
    mat_name = mat_name,
    eps = eps
  )
}