#' Nine-dimensional "fuzzy" simplex
#'
#' A synthetic data set, consisting of a "fuzzy" nine-dimensional simplex: ten
#' points equidistant from each other (the length being 2). Each point in the
#' simplex has a separate label, "0" to "9".
#'
#' Then for each vertex of the simplex, a further 99 points were generated,
#' sampled from a nine-dimensional Gaussian distribution centered at the
#' vertex, with a standard deviation of 0.5. Each of the points so generated
#' was given the same label as their "parent" vertex. This generated a
#' nine-dimensional dataset with 1000 instances and ten classes.
#'
#' This data set is intended to fulfil the following criteria:
#' \enumerate{
#'  \item Not impossibly difficult: there's reasonable overlap of the
#' ten clusters of points, but the variance is isotropic and identical for each
#' cluster.
#'  \item Have an obvious right answer by visual inspection of the output map:
#'  do we see ten reasonably well separated blobs?
#'  \item Be sufficiently complex so that the "crowding problem" will
#'  manifest: in the original nine-dimensional input space, the ten classes are
#'  by definition equidistant from each other, so it's impossible for the input
#'  to be perfectly reproduced in the two-dimensional output map.
#'  \item Traditional distance-preserving mapping methods (e.g. PCA, MDS,
#'  Sammon mapping) shouldn't do a very good job, otherwise there's no point
#'  using a probability-based method.
#' }
#'
#' The variables are as follows:
#' \itemize{
#' \item \code{D0}, \code{D1}, \code{D2} ... \code{D8} Real values, ranging
#' from -2.51 to 3.27.
#' \item \code{Label} The id of the simplex vertex that this point is
#' associated with, in the range 0-9. Stored as a factor.
#' }
#'
#' @docType data
#' @keywords datasets
#' @name s1k
#' @usage data(s1k)
#' @format A data frame with 1000 rows and 10 variables
"s1k"
