# Simulation data sets.

#' Sphere data set.
#'
#' Simulation data.
#'
#' Creates a series of points sampled from a 3D spherical surface.
#'
#' Labels go from \code{0} to \code{nlabels - 1}. Points are classified into
#' labels based on the angle theta in their spherical coordinate presentation.
#' You can think of it as dividing the sphere surface like the segments of an
#' orange.
#'
#' @param n The number of points to create.
#' @param nlabels Number of labels.
#' @return Data frame.
#' @references
#' A dataset like this was used in:
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
sphere <- function(n = 1000, nlabels = 5) {
  # from http://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
  z <- runif(n, min = -1, max = 1)
  theta <- runif(n, min = -pi, max = pi)
  x <- sin(theta) * sqrt(1 - z ^ 2)
  y <- cos(theta) * sqrt(1 - z ^ 2)
  Label <- as.factor(floor(((theta + pi) / (2 * pi)) * nlabels))
  data.frame(x, y, z, Label)
}

#' Ball Data Set
#'
#' Simulation data.
#'
#' Creates a series of points sampled from a 3D spherical volume.
#'
#' Labels go from \code{0} to \code{nlabels - 1}. Points are classified into
#' labels based on their distance from the origin, where that distance is
#' divided into \code{nlabels} equal lengths. That results in unequal
#' volumes and hence unequal populations of the labels.
#'
#' @param n Number of points to create.
#' @param rad Radius of the ball.
#' @param nlabels Number of labels.
#' @return Data frame.
#' @references
#' A dataset like this was used in:
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
ball <- function(n = 1000, rad = 1, nlabels = 5) {
  # from http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
  u <- runif(n)
  xyz <- matrix(rnorm(3 * n), nrow = n,
                dimnames = list(NULL, c("x", "y", "z")))
  df <- data.frame((xyz * rad * u ^ (1 / 3)) / sqrt(rowSums(xyz ^ 2)))
  df$Label <- as.factor(round(sqrt(rowSums(df ^ 2)) * nlabels))
  df
}

#' Toroidal Helix Data Set
#'
#' Simulation data.
#'
#' Creates a series of points sampled from a 3D helix with the ends joined
#' to each other.
#'
#' Unlike \code{\link{ball}} and \code{\link{sphere}}, this data set is not
#' randomly sampled.
#'
#' Labels go from \code{0} to \code{nlabels - 1}. Points are classified into
#' labels based on their z-coordinate, where that distance is divided into
#' \code{nlabels} equal lengths. That results in unequal volumes and hence
#' unequal populations of the labels.
#'
#' @param n Number of points to create.
#' @param rmajor Major radius.
#' @param rminor Minor radius.
#' @param nwinds Number of winds the helix makes.
#' @param nlabels Number of labels.
#' @return Data frame.
#' @references
#' A dataset like this was used in:
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
helix <- function(n = 1000, rmajor = 2, rminor = 1, nwinds = 8, nlabels = 5) {
  # http://math.stackexchange.com/questions/324527/do-these-equations-create-a-helix-wrapped-into-a-torus

  u <- seq(-pi, pi, length.out = n)

  w <- rmajor + (rminor * cos(nwinds * u))
  x <- w * cos(u)
  y <- w * sin(u)
  z <- rminor * sin(nwinds * u)

  Label <- as.factor(floor(((z + rminor) / (2 * rminor)) * nlabels))
  data.frame(x, y, z, Label)
}

#' d-Dimensional Gaussian Data Set
#'
#' Simulation data.
#'
#' Creates a series of points randomly sampled from an d-Dimensional Gaussian
#' with mean zero and variance 1.
#'
#' Labels go from \code{0} to \code{nlabels - 1}. Points are classified into
#' labels based on their distance from the origin, where that distance is
#' divided into \code{nlabels} equal lengths. That results in unequal volumes
#' and hence unequal populations of the labels.
#'
#' @param n Number of points to create.
#' @param d Dimension of the Gaussian.
#' @param nlabels Number of labels.
#' @return Data frame.
gauss <- function(n = 1000, d = 3, nlabels = 6) {
 m <- matrix(rnorm(n * d), ncol = d)
 dist <- sqrt(rowSums(m * m))
 Label <- as.factor(round((nlabels - 1) * (dist / max(dist))))
 data.frame(m, Label)
}
