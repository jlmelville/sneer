# Functions to help with visualizing embedding during the optimization
# procedure.

#' Create a label function for 2D embedding plot.
#'
#' Even quite short labels can create a crowded looking embedding plot.
#' Use this function to only print the first few characters of each label.
#'
#' @param num_label_chars The number of characters to plot from the label
#' for each data point.
#' @return a function which can be passed to the \code{label_fn} parameter
#' of the \code{make_plot} function.
make_label_fn <- function(num_label_chars = 1) {
  partial(substr, start = 0, stop = num_label_chars)
}

#' Create a plotting function.
#'
#' Factory function for a plotting callback which can be used by the epoch
#' function of an embedding to plot the current (two-dimensional embedding).
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
make_plot <- function(x, attr_name,
                      label_fn = function(labels) {
                        labels
                      },
                      mat_name = "ym") {
  attr <- x[[attr_name]]
  uniq_attr <- sort(unique(attr))
  colors <- rainbow(length(uniq_attr))
  names(colors) <- uniq_attr
  function(out) {
    plot(out[[mat_name]], type = "n", xlab = "D1", ylab = "D2")
    text(out[[mat_name]], labels = label_fn(attr), col = colors[attr])
  }
}

#' Helper function for visualizing the iris data set. If embedding the iris
#' data set, the result of invoking this can be passed to the \code{plot_fn}
#' parameter of the \code{make_epoch} function.
#'
#' @param num_label_chars The number of characters to plot from the label
#' for each data point.
#' @return Function for plotting the embedded iris data set.
make_iris_plot <- function(num_label_chars = 1) {
  make_plot(iris, "Species", make_label_fn(num_label_chars))
}
