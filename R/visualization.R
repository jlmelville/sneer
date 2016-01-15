# Functions to help with visualizing embedding during the optimization
# procedure.

#' Embedding Plots
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
#'  embed_prob(reporter = make_reporter(report_every = 100,
#'                                     normalize_cost = TRUE,
#'                                     plot = make_plot(iris, "Species")),
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

#' Embedding Plot Using \code{graphics} Library
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
#' tsne_iris <- embed_prob(iris[, 1:4])
#' # view the TSNE embedding
#' iris_plot(tsne_iris$ym)
#'}
make_embedding_plot <- function(x, attr_name, label_fn) {
  attr <- x[[attr_name]]
  uniq_attr <- sort(unique(attr))
  colors <- rainbow(length(uniq_attr))
  names(colors) <- uniq_attr
  if (!is.null(label_fn)) {
    labels <- label_fn(attr)
  } else {
    labels <- attr
  }
  function(ym) {
    plot(ym, type = "n", xlab = "D1", ylab = "D2")
    text(ym, labels = labels, col = colors[attr])
  }
}

#' Labels for 2D Embedding Plot
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

#' Embedding Plot for Iris Dataset
#'
#' Creates a function which can be used to visualize embeddings of the iris
#' dataset.
#'
#' Wrapper function for visualizing the iris data set. If embedding the iris
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

#' Embedding Plot Using \code{ggplot2} Library
#'
#' Creates a function which can be used to visualize embeddings from sneer
#' output results for a particular dataset using ggplot2.
#'
#' @note Use of this function requires that the \code{ggplot2} package be
#' installed.
#'
#' @param x Data frame containing label information for the embedded data.
#' @param attr_name Name of the label column in \code{x}.
#' @param mat_name The name of the matrix containing the embedded data in the
#' output list \code{out} which will be passed to the plot function.
#' @return Function with signature \code{plot_fn(out)} where \code{out} is
#' a return value from a sneer embedding function. On invocation, the
#' data will be plotted.
#' @examples
#' \dontrun{
#' mds_iris <- embed_dist(iris[, 1:4])
#' iris_view <- make_qplot(iris, "Species")
#' iris_view(mds_iris)
#' }
#' @export
make_qplot <- function(x, attr_name, mat_name = "ym") {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("qplot function requires 'ggplot2' package")
  }
  embedding_plot <- make_embedding_qplot(x, attr_name)

  function(out) {
      embedding_plot(out[[mat_name]])
  }
}

#' Embedded Coordinates Plot Using \code{ggplot2} Library
#'
#' Creates a ggplot2 function for embeddings of a given dataset.
#'
#' @note Use of this function requires that the \code{ggplot2} package be
#' installed.
#'
#' @param x Data frame containing label information for the embedded data.
#' @param attr_name Name of the label column in \code{x}.
#' @return Function with signature \code{plot_fn(ym)} where \code{ym} is a
#' 2D matrix of embedded coordinates of data set \code{x}. On invocation, the
#' data will be plotted.
#' @examples
#' \dontrun{
#' # make two different embeddings of the iris dataset
#' prcomp_iris <- prcomp(iris[, 1:4], center = TRUE, retx = TRUE)
#' mds_iris <- embed_dist(iris[, 1:4], method = mmds(eps = 1e-4),
#'                        opt = bold_nag(),
#'                        init_out = out_from_matrix(prcomp_iris$x[, 1:2]),
#'                        max_iter = 40)
#' iris_view <- make_embedding_qplot(iris, "Species")
#' iris_view(prcomp_iris$x)
#' iris_view(mds_iris$ym)
#' }
make_embedding_qplot <- function(x, attr_name) {
  function(ym) {
    if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
      stop("qplot function requires 'ggplot2' package")
    }
    colnames(ym) <- NULL
    df <- as.data.frame(ym)
    print(
      ggplot2::qplot(df$V1, df$V2, data = df,
                     colour = factor(x[[attr_name]]), size = I(3)) +
        ggplot2::scale_colour_brewer(palette = "Set3") +
        ggplot2::theme_dark() +
        ggplot2::labs(colour = attr_name)
    )
  }
}
