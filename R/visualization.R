# Functions to help with visualizing embedding during the optimization
# procedure.

#' Create a plotting function.
#'
#' Factory function for a plotting callback which can be used by the reporter
#' function of an embedding to plot the current (two-dimensional) embedding.
#'
#' @param x Data frame containing label information for the embedded data.
#' @param attr_name Name of the label column in \code{x}.
#' @param label_fn Function with the signature \code{label_fn(labels)} where
#' \code{labels} is a vector of labels for each point in the data set, and
#' the function returns a vector of labels suitable for displaying in the plot
#' (e.g. with each label truncated so that only the first three characters)
#' will be used.
#' @param mat_name The name of the matrix containing the embedded data in the
#' output list \code{out} which will be passed to the plot function.
#' @return Function which will take an output list, and produce a 2D plot of
#' the embedding.
#' @seealso \code{\link{make_reporter}} for how to use this function for
#' configuring visualization of the progress of an embedding.
#' @examples
#' # For s1k dataset, plot 2D embedding with "Label" factor to identify each
#' # point on the plot
#' make_plot(s1k, "Label")
#'
#' # For iris dataset, plot 2D embedding with first two characters of the
#' # "Species" factor to identify each point on the plot
#' make_plot(iris, "Species", make_label(2))
#'
#' # Should be passed to the plot_fn argument of the reporter factory function:
#' \dontrun{
#'  embed_sim(reporter = make_reporter(report_every = 100,
#'                                     normalize_cost = TRUE,
#'                                     plot_fn = make_plot(iris, "Species")),
#'                                      ...)
#' }
#' @family sneer plot functions
#' @export
make_plot <- function(x, attr_name,
                      label_fn = function(labels) {
                        labels
                      },
                      mat_name = "ym") {
  embedding_plot <- make_embedding_plot(x, attr_name, label_fn)
  function(out) {
    embedding_plot(out[[mat_name]])
  }
}

#' Create embedding plot.
#'
#' Create a function which when invoked on a 2D matrix, plots the embedding
#' with color-coded labels.
#'
#' @param x Data frame containing label information for the embedded data.
#' @param attr_name Name of the label column in \code{x}.
#' @param label_fn Function with the signature \code{label_fn(labels)} where
#' \code{labels} is a vector of labels for each point in the data set. The
#' function should return a vector of labels suitable for displaying in the
#' plot.
#' @return Function which will take an output list, and produce a 2D plot of
#' the embedding.
#' @seealso \code{\link{make_label}} for an example of a suitable
#' argument to \code{label_fn}.
#'
#' @examples
#' \dontrun{
#' # Create a plot function for the Iris dataset
#' iris_plot <- make_embedding_plot(iris, "Species")
#'
#' # PCA on iris
#' pca_iris <- prcomp(iris[, 1:4], center = TRUE, retx = TRUE)
#' # view the first two scores:
#' iris_plot(pca_iris$x[, 1:2])
#'
#' # TSNE on iris
#' tsne_iris <- embed_sim(iris[, 1:4], method = tsne())
#' # view the TSNE embedding
#' iris_plot(tsne_iris$ym)
#'}
#'@export
make_embedding_plot <- function(x, attr_name,
                               label_fn = function(labels) {
                                labels
                               }) {
  attr <- x[[attr_name]]
  uniq_attr <- sort(unique(attr))
  colors <- rainbow(length(uniq_attr))
  names(colors) <- uniq_attr
  function(ym) {
    plot(ym, type = "n", xlab = "D1", ylab = "D2")
    text(ym, labels = label_fn(attr), col = colors[attr])
  }
}

#' Create a label function for 2D embedding plot.
#'
#' Even quite short labels can create a crowded looking embedding plot.
#' Use this function to only print the first few characters of each label.
#'
#' @param num_label_chars The number of characters to plot from the label
#' for each data point.
#' @return a function which can be passed to the \code{label_fn} parameter
#' of the \code{make_plot} function.
#' @export
make_label <- function(num_label_chars = 1) {
  partial(substr, start = 0, stop = num_label_chars)
}

#' Helper function for visualizing the iris data set. If embedding the iris
#' data set, the result of invoking this can be passed to the \code{plot_fn}
#' parameter of the \code{make_reporter} function.
#'
#' @param num_label_chars The number of characters to plot from the label
#' for each data point.
#' @return Function for plotting the embedded iris data set.
#' @family sneer plot functions
#' @export
make_iris_plot <- function(num_label_chars = 1) {
  make_plot(iris, "Species", make_label(num_label_chars))
}
