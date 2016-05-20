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
#' @export
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
#' Labels go from \code{1} to \code{nlabels}. Points are classified into
#' equally populated labels based on their distance from the origin.
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
#' @export
ball <- function(n = 1000, rad = 1, nlabels = 5) {
  # from http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
  u <- runif(n)
  xyz <- matrix(rnorm(3 * n), nrow = n,
                dimnames = list(NULL, c("x", "y", "z")))
  df <- data.frame((xyz * rad * u ^ (1 / 3)) / sqrt(rowSums(xyz ^ 2)))
  df$Label <- equal_factors(sqrt(rowSums(df ^ 2)), nlabels)
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
#' Labels go from \code{1} to \code{nlabels}. Points are classified into
#' equally-populated labels based on their z-coordinate.
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
#' @export
helix <- function(n = 1000, rmajor = 2, rminor = 1, nwinds = 8, nlabels = 5) {
  # http://math.stackexchange.com/questions/324527/do-these-equations-create-a-helix-wrapped-into-a-torus
  u <- seq(-pi, pi, length.out = n)
  w <- rmajor + (rminor * cos(nwinds * u))
  x <- w * cos(u)
  y <- w * sin(u)
  z <- rminor * sin(nwinds * u)

  Label <- equal_factors(z, nlabels)
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
#' @export
gauss <- function(n = 1000, d = 3, nlabels = 5) {
  m <- matrix(rnorm(n * d), ncol = d)
  dist <- sqrt(rowSums(m * m))

  Label <- equal_factors(dist, nlabels)

  data.frame(m, Label)
}

#' Swiss Roll Data Set
#'
#' Simulation data.
#'
#' Creates a series of points randomly sampled from a swiss roll-shaped
#' manifold: a two-dimensional plane which has been rolled up into a spiral
#' shape. Or just look at a swiss roll.
#'
#' The formula for sampling the x, y and z coordinates used in this dataset is
#' from that given in the Stochastic Proximity Embedding paper by Agrafiotis
#' and Xu:
#' \deqn{x = \phi cos\phi, y = \phi sin \phi, z}
#' {x = phi * cos(phi), y = phi * sin(phi), z}
#'
#' where \eqn{\phi}{phi} and z are random numbers in the intervals [5, 13]
#' and [0, 10], respectively.
#'
#' Labels go from \code{0} to \code{nlabels - 1}. Points are classified into
#' labels based on their z-coordinate.
#'
#' @param n Number of points to create.
#' @param nlabels Number of labels.
#' @return Data frame.
#' @references
#' Agrafiotis, D. K., & Xu, H. (2002).
#' A self-organizing principle for learning nonlinear manifolds.
#' \emph{Proceedings of the National Academy of Sciences}, \emph{99}(25), 15869-15872.
#' @export
swiss_roll <- function(n = 1000, nlabels = 6) {
  phi <- runif(n, min = 5, max = 30)
  x <- phi * cos(phi)
  y <- phi * sin(phi)
  z <- runif(n, max = 10)

  Label <- as.factor(ceiling(((phi - 5) / 25) * nlabels - 1))

  data.frame(x, y, z, Label)
}

#' Frey Faces dataset
#'
#' Sneer benchmarking data.
#'
#' Returns the Frey Faces dataset in a data frame reformatted to be suitable
#' for use with Sneer. This is a series of 1965 images (with dimension 20 x 28)
#' of Brendan Frey's face taken from sequential frames of a video.
#'
#' The variables are as follows:
#' \itemize{
#' \item \code{px1}, \code{px2}, \code{px3} ... \code{px560} 8-bit grayscale
#' pixel values (0-255). The pixel index starts at the top right of the image
#' (\code{px1}) and are then stored row-wise.
#' \item \code{Label} An integer in the range (1-1965) indicating the frame.
#' Provides the same information as the row id, but included to be consistent
#' with other sneer datasets.
#' }
#'
#' @note requires the \code{RnavGraphImageData} package.
#' \url{https://cran.r-project.org/web/packages/RnavGraphImageData/index.html}
#' to be installed and loaded.
#' @return the Frey Faces dataset as a dataframe.
#' @format A data frame with 1965 rows and 561 variables.
#' @seealso
#' Saul Roweis' dataset web page: \url{http://www.cs.nyu.edu/~roweis/data.html}.
#' Each row can be visualized as an image using \code{\link{show_frey_face}}.
#' @export
frey_faces <- function() {
  if (!requireNamespace("RnavGraphImageData", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("frey_faces function requires 'RnavGraphImageData' package")
  }
  frey <- NULL
  data("frey", envir = environment())

  df <- data.frame(t(frey))
  colnames(df) <- sapply(seq(1, 20 * 28), function(x) { paste0("px", x)})

  df$Label <- factor(as.numeric(cut(1:nrow(df), nrow(df))))
  df
}

#' Visualize Frey face.
#'
#' Display an image from the Frey faces dataset.
#'
#' @param df Data frame containing the Frey faces.
#' @param n Frame index of the image to display.
#' @param col List of colors to use in the display.
#' @export
show_frey_face <- function(df, n, col = gray(1 / 12:1)) {
  if (n < 1 || n > nrow(df)) {
    stop("n must be value between 1 and ", nrow(df))
  }
  im <- matrix(t(df[n, 560:1]), ncol = 28, nrow = 20)
  image(1:nrow(im), 1:ncol(im), im, xlab = "", ylab = "", col = col)
}

#' Olivetti Faces dataset
#'
#' Sneer benchmarking data.
#'
#' Returns the Olivetti Faces dataset in a data frame reformatted to be suitable
#' for use with Sneer. This is a series of 400 images (with dimension 64 x 64)
#' of 40 individual's faces, with ten different poses per person.
#'
#' The variables are as follows:
#' \itemize{
#' \item \code{px1}, \code{px2}, \code{px3} ... \code{px560} 8-bit grayscale
#' pixel values (0-255). The pixel index starts at the top right of the image
#' (\code{px1}) and are then stored column-wise.
#' \item \code{Label} An integer in the range (1-40) indicating the person.
#' }
#'
#' Each row has a name with the format "<face>_<pose>", where \code{<face>} is
#' the index of the face, and \code{<pose>} is the index of the pose, e.g.
#' the row with name \code{20_10} is the tenth pose of the twentieth face.
#'
#' @note requires the \code{RnavGraphImageData} package.
#' \url{https://cran.r-project.org/web/packages/RnavGraphImageData/index.html}
#' to be installed and loaded.
#' @return the Olivetti Faces dataset as a dataframe.
#' @format A data frame with 400 rows and 4097 variables.
#' @seealso
#' Saul Roweis' dataset web page: \url{http://www.cs.nyu.edu/~roweis/data.html}.
#' Each row can be visualized as an image using
#' \code{\link{show_olivetti_face}}.
#' @export
olivetti_faces <- function() {
  if (!requireNamespace("RnavGraphImageData", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("olivetti_faces function requires 'RnavGraphImageData' package")
  }
  faces <- NULL
  data("faces", envir = environment())

  df <- as.data.frame(t(faces))
  npeople <- 40
  nposes <- 10
  colnames(df) <- sapply(seq(1, 4096), function(x) { paste0("px", x)})
  rownames(df) <- apply(expand.grid(seq(1, nposes), seq(1, npeople)), 1,
        function(x) { paste(x[2], x[1], sep = "_") })
  df$Label <-  factor(as.numeric(cut(1:nrow(df), npeople)))

  df
}

#' Visualize Olivetti face.
#'
#' Display an image from the Olivetti faces dataset.
#'
#' @param df Data frame containing the Olivetti faces.
#' @param face Face index of the image to display. Must be an integer between
#' 1 and 400.
#' @param pose Pose index of the image to display. Must be an integer between
#' 1 and 10.
#' @param col List of colors to use in the display.
#' @export
show_olivetti_face <- function(df, face, pose, col = gray(1 / 12:1)) {
  if (face < 1 || face > 400) {
    stop("face must be an integer between 1 and 400")
  }
  if (pose < 1 || pose > 10) {
    stop("pose must be an integer between 1 and 10")
  }
  n <- paste(face, pose, sep = "_")
  im <- t(matrix(as.numeric(df[n, 4096:1]), ncol = 64, nrow = 64))
  image(1:nrow(im), 1:nrow(im), im, xlab = "", ylab = "", col = col)
}

# Split A Vector Into Equally Populated Factors
#
# Assigns each member of a vector to a factor, based on the quantiles of the
# distribution, so that each factor is equal populated. Levels range from
# one to \code{nfactors}.
#
# @param x Numeric vector
# @param nfactors Number of factors required.
# @return factor-encoded vector specifying the level for each item in the
# vector.
equal_factors <- function(x, nfactors) {
  breaks <- quantile(x, probs = seq(0, 1, length.out = nfactors + 1))
  cuts <- cut(x, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  as.factor(cuts)
}
