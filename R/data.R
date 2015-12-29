#' Nine-dimensional simplex data set.
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
#'  \item not impossibly difficult: there's reasonable overlap of the
#' ten clusters of points, but the variance is isotropic and identical for each
#' cluster.
#'  \item have an obvious right answer by visual inspection of the output map:
#'  do we see ten reasonably well separated blobs?
#'  \item to be sufficiently complex so that the "crowding problem" will
#'  manifest: in the original nine-dimensional input space, the ten classes are
#'  by definition equidistant from each other, so it's impossible for the input
#'  to be perfectly reproduced in the two-dimensional output map.
#'  \item traditional distance-preserving mapping methods (e.g. PCA, MDS,
#'  Sammon mapping) shouldn't do a very good job, otherwise there's no point
#'  using a probability-based method.
#' }
#' @docType data
#'
#' @usage data(s1k)
#'
#' @keywords datasets
"s1k"
