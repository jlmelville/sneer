# Functions to help with visualizing embedding during the optimization
# procedure.

#' Embedding Plot, Colored By Category
#'
#' Plots the embedded coordinates, with each point colored by a specified
#' color.
#'
#' The \code{x} argument can be used to provide a suitable vector of colors
#' from either a data frame or vector.
#'
#' If a data frame is provided, then a vector of colors will be looked for. If
#' it's present, it will be used as the \code{colors} argument directly.
#' Otherwise, a factor column will be looked for, and each level will be mapped
#' to a different color. Otherwise, one color will be used for each point. If
#' more than one column of a type is found in the data frame, the last one
#' encountered is used.
#'
#' If a vector is provided, a similar procedure to the data frame is used when
#' mapping from its content to a vector of colors. Additionally, a numeric vector
#' can be provided, which will be linearly mapped to a color scheme.
#'
#' The \code{color_scheme} parameter can be one of either a color ramp function,
#' accepting an integer n as an argument and returning n colors, or the name of
#' a ColorBrewer color scheme. Probably should be one of the "Qualitative" set.
#'
#' For some applicable color ramp functions, see the \code{Palettes} help page
#' in the \code{grDevices} package (e.g. by running the \code{?rainbow} command).
#'
#' @param coords Matrix of embedded coordinates, with as many rows as
#'  observations, and 2 columns.
#' @param colors Vector containing colors for each coordinate.
#' @param x Either a data frame or a column that can be used to derive a
#'  suitable vector of colors. Ignored if \code{colors} is provided.
#' @param color_scheme Either a color ramp function, or the name of a
#'  ColorBrewer scheme. See 'Details'.
#' @param num_colors Number of unique colors to map to from \code{x}, if
#'  \code{x} is a numeric vector. Otherwise ignored.
#' @param limits The range that the colors should map over when mapping from a
#'  numeric vector. If not specified, then the range of \code{x}. This is useful
#'  if there is some external absolute scale that should be used. Ignored if
#'  \code{x} is not a numeric vector.
#' @param top If not \code{NULL}, only the specified number of points will be
#'  displayed, corresponding to those with the highest values in \code{vec},
#'  after sorting by decreasing order.
#'  num_colors = 15, limits = NULL, top = NULL,
#' @param cex Size of the points. Ignored if \code{as_text} is \code{TRUE}.
#' @param text Vector of label text to display instead of a point. If the labels
#'  are long or the data set is large, this is unlikely to be very legible, but
#'  is occasionally useful.
#' @param equal_axes If \code{TRUE}, the X and Y axes are set to have the
#'  same extents.
#' @note Use of this function with ColorBrewer qualitative palette names
#' requires that the \code{RColorBrewer} package be installed.
#'
#' More information on ColorBrewer is available at its website,
#'  \url{http://www.colorbrewer2.org}.
#' @export
#' @examples
#' \dontrun{
#' # Embed with PCA
#' pca_iris <- sneer(iris, method = "pca", scale_type = "a", ret = c("dy"))
#' # Visualize the resulting embedding, colored by iris species, using the
#' # rainbow color scheme
#' embed_plot(pca_iris$coords, x = iris$Species, color_scheme = rainbow)
#'
#' # topo.colors scheme
#' embed_plot(pca_iris$coords, x = iris$Species, color_scheme = topo.colors)
#'
#' # Pass in data frame and it will use any factor column it finds
#' embed_plot(pca_iris$coords, x = iris)
#'
#'#' library("RColorBrewer")
#' # Use the "Dark2" ColorBrewer scheme
#' embed_plot(pca_iris$coords, x = iris, color_scheme = "Dark2")
#'
#' # Can plot the category names instead of points, but looks bad if they're
#' # long (or the dataset is large)
#' embed_plot(pca_iris$coords, x = iris$Species, cex = 0.5, as_text = TRUE)
#'
#' tsne_iris <- sneer(iris, method = "tsne", scale_type = "a",
#'                    ret = c("dx", "dy", "deg"))
#' # how well is the 32 nearest neighborhood preserved for each point?
#' nbr_pres_32 <- nbr_pres(tsne_iris$dx, tsne_iris$dy, 32)
#' # visualize preservation, use absolute scale of 0-1 for colors.
#' embed_plot(tsne_iris$coords, x = nbr_pres_32, limits = c(0, 1))
#'
#' # visualize 10 points with the hightest degree centrality
#' embed_plot(tsne_iris$coords, x = tsne_iris$deg, top = 10)
#' }
embed_plot <- function(coords, colors = NULL, x = NULL,
                       color_scheme = grDevices::rainbow,
                       num_colors = 15, limits = NULL, top = NULL,
                       cex = 1, text = NULL, equal_axes = FALSE) {

  if (is.null(colors)) {
    if (!is.null(x)) {
      colors <- color_helper(x, color_scheme = color_scheme,
                             num_colors = num_colors, limits = limits,
                             top = top)
    }
    else {
      colors <- make_palette(ncolors = nrow(coords),
                             color_scheme = color_scheme)
    }
  }

  lims <- NULL
  if (equal_axes) {
    lims <- range(coords)
  }

  if (!is.null(text)) {
    graphics::plot(coords, type = 'n', xlim = lims, ylim = lims,
                   xlab = 'X', ylab = 'Y')
    graphics::text(coords, labels = text, cex = cex, col = colors)
  }
  else {
    graphics::plot(coords, pch = 20, cex = cex, col = colors,
                   xlim = lims, ylim = lims, xlab = 'X', ylab = 'Y')
  }
}

# Helper function for generating color (and maybe factor) vectors from a data
# frame or distance matrix, df, for use in an embedding plot, via multiple
# possible avenues.
#
# Users may provide any of the following, which are tested in the following
#  order. As soon as one succeeds, that's the column that will be used.
# colors - provide a color vector explicitly, in which case just use that.
# color_name - the name of a column in df that contains colors.
# labels - a column of factors which will be mapped to colors.
# label_name - the name of a column in df that contains a factor to be mapped
#   to colors.
# color_scheme - either a color ramp function of an RColorBrewer color scheme
#   name to use in the mapping of labels to colors.
# If nothing is provided, we look in df directly for a color column. If there's
# more than one, we use the last column found. If there are no color columns,
# we look for a factor column. If more than one is found, we use the last column
# found.
#
# If df is a distance matrix, then the label_name and color_name parameters
# are ignored.
#
# A list is returned containing:
#   colors - the colors that are going to be used
#   labels - if label_name was used or we found a factor column ourselves, this
#            is the vector of labels that was used. Necessary for
#            plotting the labels as text in an embedding plot.
process_color_options <- function(df,
                                  colors = NULL,
                                  color_name = NULL,
                                  labels = NULL,
                                  label_name = NULL,
                                  color_scheme = grDevices::rainbow,
                                  verbose = FALSE) {
  if (is.null(colors)) {
    # if no color vector was provided, look for a color name
    if (!is.null(color_name) && class(df) == "data.frame") {
      colors <- df[[color_name]]
      if (is.null(colors)) {
        stop("Couldn't find color column '", color_name, "'")
      }
      if (!is_color_column(colors)) {
        stop("Column '", color_name, "' does not contain colors")
      }
    }
  }

  if (is.null(colors)) {
    # Neither colors nor color_name was specified, let's try with labels
    if (is.null(labels) && !is.null(label_name) && class(df) == "data.frame") {
      # No labels provided, but there was a label name
      labels <- df[[label_name]]
      if (is.null(labels)) {
        stop("Couldn't find label column '", label_name, "'")
      }
    }
    if (!is.null(labels)) {
      # Either we provided explicit labels or the label name worked out
      if (class(labels) != "factor") {
        stop("Label column should be a factor")
      }
      colors <- factor_to_colors(labels, color_scheme = color_scheme)
    }
  }
  # Neither labels nor colors provided (or names to look up)
  # Let's go look ourselves
  if (is.null(colors) && class(df) == "data.frame") {
    res <- color_helper_df(df = df, color_scheme = color_scheme,
                           ret_labels = TRUE, verbose = verbose)
    if (is.null(colors)) {
      colors <- res$colors
    }
    if (is.null(labels)) {
      labels <- res$labels
    }
  }
  list(colors = colors, labels = labels)
}

# Given a data frame or a vector, return a vector of colors appropriately
# mapped to the color scheme.
# If \code{x} is a vector, it can either be a vector of colors, a factor
# vector (in which case each level is mapped to a color), or a numeric vector
# (in which case the range is mapped linearly).
# If \code{x} is a data frame, then it is checked for a color column. If there
# isn't one, a factor column is looked for. If there's more than one suitable
# column, the last found column is used. Numeric columns aren't searched for in
# the data frame case.
color_helper <- function(x,
                        color_scheme = grDevices::rainbow,
                        num_colors = 15, limits = NULL, top = NULL,
                        verbose = FALSE) {
  if (class(x) == 'data.frame') {
    color_helper_df(x, color_scheme = color_scheme, ret_labels = FALSE,
                    verbose = verbose)
  }
  else {
    color_helper_column(x,
                        color_scheme = color_scheme,
                        num_colors = num_colors, limits = limits, top = top,
                        verbose = verbose)
  }
}

# Try and find a meaningful vector of colors from a data frame.
# If the data frame contains at least one column of colors, use the last column
# of colors found.
# Otherwise, if the data frame contains at least one column of factors, map
# from the last factor column found to a list of colors.
# Otherwise, color each point as if it was its own factor level
# @note R considers numbers to be acceptable colors because \code{col2rgb}
# interprets them as indexes into a palette. Columns of numbers are NOT treated
# as colors by color_helper. Stick with color names (e.g. "goldenrod") or
# rgb strings (e.g. "#140000" or "#140000FF" if including alpha values).
color_helper_df <- function(df,
                            color_scheme = color_scheme,
                            ret_labels = FALSE,
                            verbose = FALSE) {

  colors <- NULL
  labels <- NULL
  # Is there a color column?
  color_name <- last_color_column_name(df)
  if (!is.null(color_name)) {
    if (verbose) {
      message("Found color column ", color_name)
    }
    colors <- df[[color_name]]
  }

  if (is.null(colors)) {
    # Is there a factor column?
    label_name <- last_factor_column_name(df)
    if (!is.null(label_name)) {
      if (verbose) {
        message("Found a factor ", label_name, " for mapping to colors")
      }
      labels <- df[[label_name]]
      colors <- factor_to_colors(labels, color_scheme = color_scheme)
    }
  }

  if (is.null(colors)) {
    # use one color per point
    colors <- make_palette(ncolors = nrow(df), color_scheme = color_scheme)
  }

  # Return a list with both results if we want labels, otherwise just colors
  if (ret_labels) {
    res <- list(colors = colors, labels = labels)
  }
  else {
    res <- colors
  }
  res
}

color_helper_column <- function(x,
                                color_scheme = color_scheme,
                                num_colors = 15, limits = NULL, top = NULL,
                                verbose = FALSE) {
  # Is this a color column - return as-is
  if (is_color_column(x)) {
    return(x)
  }

  # Is it numeric - map to palette (which should be sequential or diverging)
  if (is.numeric(x)) {
    colors <- numeric_to_colors(x, color_scheme = color_scheme,
                                n = num_colors, limits = limits)
    if (!is.null(top)) {
      svec <- sort(x, decreasing = TRUE)
      colors[x < svec[top]] <- NA
    }
    return(colors)
  }

  # Is it a factor - map to palette (which should be categorical)
  if (is.factor(x)) {
    return(factor_to_colors(x, color_scheme = color_scheme))
  }

  # Otherwise one color per point (doesn't really matter what the palette is!)
  make_palette(ncolors = length(x), color_scheme = color_scheme)
}


# Map a vector of factor levels, x,  to a vector of colors taken from either
# an RColorBrewer palette name, or a color ramp function.
# @examples
# factor_to_colors(iris$Species, color_scheme = "Set3") # ColorBrewer palette
# factor_to_colors(iris$Species, color_scheme = rainbow) # color ramp function
factor_to_colors <- function(x, color_scheme = grDevices::rainbow) {
  category_names <- unique(x)
  ncolors <- length(category_names)
  color_palette = make_palette(ncolors = ncolors, color_scheme = color_scheme)
  color_palette[x]
}


# Map Numbers to Colors
#
# Maps a numeric vector to an equivalent set of colors based on the specified
# ColorBrewer palette. Use the diverging or sequential.
#
# Sequential palettes names:
#  Blues BuGn BuPu GnBu Greens Greys Oranges OrRd PuBu PuBuGn PuRd Purples
#  RdPu Reds YlGn YlGnBu YlOrBr YlOrRd
# Diverging palette names:
#  BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
#
# @note Use of this function requires that the \code{RColorBrewer} packages be
#  installed.
# @note This function is based off a Stack Overflow answer by user "Dave X":
#  \url{http://stackoverflow.com/a/18749392}
#
# @param x Numeric vector.
# @param name Name of the ColorBrewer palette.
# @param n Number of unique colors to map values in \code{x} to.
# @param limits The range that the colors should map over. If not specified,
#  then the range of \code{x}. This is useful if there is some external
#  absolute scale that should be used.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
# @examples
# \dontrun{
# # Plot Iris dataset sepal width vs length, colored by petal length, using
# # 20 colors ranging from Purple to Green (PRGn):
# plot(iris[, c("Sepal.Length", "Sepal.Width")], cex = 1.5, pch = 20,
#  col = numeric_to_colors(iris$Petal.Length, palette = "PRGn", n = 20))
#
# # Use the rainbow color ramp function
# plot(iris[, c("Sepal.Length", "Sepal.Width")], cex = 1.5, pch = 20,
#  col = numeric_to_colors(iris$Petal.Length, col_ramp = rainbow, n = 20))
# }
numeric_to_colors <- function(x, color_scheme = "Blues", n = 15,
                              limits = NULL) {
  if (class(color_scheme) == "character" &&
      !requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("numeric_to_colors function requires 'RColorBrewer' package")
  }
  if (is.null(limits)) {
    limits <- range(x)
  }
  pal <- make_palette(ncolors = n, color_scheme = color_scheme)
  pal[findInterval(x, seq(limits[1], limits[2], length.out = length(pal) + 1),
                   all.inside = TRUE)]
}

# Color Palette with Specified Number of Colors
#
# Creates a color palette with the specified number of colors, interpolating
# ColorBrewer palettes by default.
#
# This function is designed to make it easy to use the ColorBrewer palettes,
# particularly with the qualitative sets, without having to worry about a plot
# not being displayed because the palette didn't have enough colors for the
# number of categories required. Admittedly, you probably shouldn't be using
# the palette in that case, but it's better to see the plot.
#
# Rather than specify a ColorBrewer scheme by name, you can also pass in
# a color ramp function of any kind. For some applicable ramp functions, see
# the \code{Palettes} help page in the \code{grDevices} package (e.g. by
# running the \code{?rainbow} command).
#
# @param ncolors Number of colors desired for the palette.
# @param color_scheme Either the name of a ColorBrewer palette, or a function
#  accepting an integer n as an argument and returning
#  n colors.
# @value A palette with the specified number of colors, interpolated if
#  necessary.
make_palette <- function(ncolors, color_scheme = grDevices::rainbow) {
  if (class(color_scheme) == "function") {
    palette <- color_scheme(ncolors)
  }
  else {
    if (!requireNamespace("RColorBrewer", quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("make_palette function requires 'RColorBrewer' package")
    }
    palette <- color_brewer_palette(color_scheme, ncolors)
  }
  palette
}



# PCA Scores Plot
#
# Carries out PCA on a data frame (using the provided indexes) and then
# plots the first two scores, coloring the points with a user-specified
# label.
#
# @param df Data frame.
# @param pca_indexes Numeric vector containing the column indexes in the
#  data frame which should be used for the PCA calculation. Default is
#  \code{NULL}, in which case all numeric columns will be used.
# @param center If \code{TRUE}, mean-center the columns.
# @param scale If \code{TRUE}, scale the columns to have unit variance.
# @param colors Vector of colors to apply to each point. If \code{NULL}, then
#  the data frame will be searched for a suitable vector of colors or a vector
#  of factor levels to map to the \code{grDevices::rainbow} color scheme.
# @param cex Numeric \strong{c}haracter \strong{ex}pansion factor;
#  multiplied by \code{\link[graphics]{par}("cex")} yields the final
#  character size of the labels.
# @param text Vector of label text to display instead of a point. If the labels
#  are long or the data set is large, this is unlikely to be very legible, but
#  is occasionally useful.
# @param verbose If \code{TRUE} log messages about any default behavior
# (e.g. number of columns used in PCA or category column)
# @examples
# \dontrun{
#  # PCA on the scaled iris dataset, use the "Species" column as labels.
#  # Change the text size to make it easier to read.
#  scores_plot(iris, 1:4, scale = TRUE, text = iris$Species, cex = 0.5)
#
#  # PCA on the s1k dataset, use all numeric values, and use default coloring
#  # (mapping factor levels to colors).
#  scores_plot(s1k)
# }
scores_plot <- function(df, pca_indexes = NULL,
                        center = TRUE, scale = FALSE,
                        colors = NULL,
                        cex = 1, text = NULL,
                        verbose = FALSE) {
  if (is.null(pca_indexes)) {
    pca_indexes <- which(vapply(df, is.numeric, logical(1)))
  }
  if (is.null(pca_indexes)) {
    stop("Couldn't find any numeric columns to carry PCA out on!")
  }
  else if (verbose) {
    message("Using ", length(pca_indexes), " columns for PCA")
  }

  pca <- stats::prcomp(df[, pca_indexes], retx = TRUE, center = center,
                       scale. = scale)

  embed_plot(pca$x[, 1:2],
             x = df,
             colors = colors,
             color_scheme = grDevices::rainbow,
             cex = cex,
             text = text)
}

# PCA Scores Plot Using ggplot2 and ColorBrewer Palettes
#
# Carries out PCA on a dataset and then displays the first two scores in a
# scatterplot using ggplot2 and a color palette from RColorBrewer.
#
# @note Use of this function requires that the \code{ggplot2} and
# \code{RColorBrewer} packages be installed.
#
# @param df Data frame.
# @param pca_indexes Numeric vector containing the column indexes in the
#  data frame which should be used for the PCA calculation. Default is
#  \code{NULL}, in which case all numeric columns will be used.
# @param label_name name of the column that contains a value to be used
#  to color each point. Default is \code{NULL}, in which case the first
#  encountered factor column will be used.
# @param center If \code{TRUE}, mean-center the columns.
# @param scale If \code{TRUE}, scale the columns to have unit variance.
# @param size Size of the points.
# @param color_scheme Color ramp function or string giving the name of a
#   ColorBrewer Palette. To see the
#  available ColorBrewer schemes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following schemes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
# @examples
#  \dontrun{
#  # PCA on the scaled iris dataset, use the "Species" column to display
#  scores_qplot(iris, 1:4, "Species", scale = TRUE)
#
#  # PCA on s1k dataset, use all numeric indices and first factor (defaults)
#  # for coloring with the "Set3" scheme, set point size to 2,
#  # and display three rows in the legend
#  scores_qplot(s1k, size = 2, color_scheme = "Set3", legend_rows = 3)
# }
scores_qplot <- function(df, pca_indexes = NULL, label_name = NULL,
                         center = TRUE, scale = FALSE,
                         size = 1,
                         color_scheme = "Set1",
                         legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("scores_qplot function requires 'ggplot2' package")
  }
  if (class(color_scheme) == "character" &&
      !requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("scores_qplot function requires 'RColorBrewer' package")
  }

  if (is.null(pca_indexes)) {
    pca_indexes <- vapply(df, is.numeric, logical(1))
  }

  if (is.null(label_name)) {
    factor_names <- names(df)[(vapply(df, is.factor, logical(1)))]
    if (length(factor_names) == 0) {
      stop("Couldn't find a factor column in data frame to use for label")
    }
    else {
      label_name <- factor_names[1]
    }
  }

  if (is.null(df[[label_name]])) {
    stop("Data frame does not have a '",label_name,
         "' column for use as a label")
  }

  pca <- stats::prcomp(df[, pca_indexes], retx = TRUE, center = center,
                       scale. = scale)
  scatterqplot(df, x = pca$x[, 1], y = pca$x[, 2],
               label_name = label_name, size = size,
               color_scheme = color_scheme,
               x_label = "t1", y_label = "t2",
               legend = legend, legend_rows = legend_rows)
}

# Scatterplot Using ggplot2 and ColorBrewer Color Schemes
#
# Scatterplot using ggplot2 with coloring via categorical data and a scheme
# from RColorBrewer.
#
# @note Use of this function requires that the \code{ggplot2} and
# \code{RColorBrewer} packages be installed.
#
# @param df Data frame.
# @param x Vector of x values to plot for the x-coordinate.
# @param y Vector of y values to plot for the y-coordinate.
# @param label_name name of the column that contains a value to be used
#  to color each point. Default is \code{NULL}, in which case the first
#  encountered factor column will be used.
# @param size Size of the points.
# @param color_scheme String giving the name of a ColorBrewer scheme To see the
#  available schemes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following schemes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param x_label Label for the x-axis.
# @param y_label Label for the y-axis.
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
scatterqplot <- function(df, x, y, label_name = NULL, labels = NULL, size = 1,
                         color_scheme = "Set1", x_label = "x", y_label = "y",
                         legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("scatterqplot function requires 'ggplot2' package")
  }
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("scatterqplot function requires 'RColorBrewer' package")
  }

  if (is.null(labels) && is.null(label_name)) {
    label_name <- last_factor_column_name(df)
    if (is.null(label_name)) {
      labels <- factor(1:nrow(df))
      legend <- FALSE
    }
  }
  if (is.null(labels)) {
    if (is.null(df[[label_name]])) {
      stop("Data frame does not have a '",label_name,
           "' column for use as a label")
    }
    labels <- df[[label_name]]
  }
  else {
    label_name <- "Labels"
  }

  ncolors <- length(unique(labels))
  if (class(color_scheme) == "function") {
    color_palette <- color_scheme(ncolors)
  }
  else {
    color_palette <- color_brewer_palette(color_scheme, ncolors)
  }

  score_plot <-
    ggplot2::qplot(x, y, colour = labels, size = I(size)) +
    ggplot2::scale_color_manual(values = color_palette, name = label_name) +
    ggplot2::theme(legend.position = "bottom",
                   panel.grid.major = ggplot2::element_blank(),
                   panel.grid.minor = ggplot2::element_blank(),
                   panel.background = ggplot2::element_blank()) +
    ggplot2::labs(x = x_label, y = y_label)


  if (!is.null(legend_rows)) {
    score_plot <- score_plot +
      ggplot2::guides(color = ggplot2::guide_legend(nrow = legend_rows))
  }
  else if (!legend) {
    score_plot <- score_plot + ggplot2::theme(legend.position = "none")
  }
  print(score_plot)
}

# Embedding Plots
#
# Factory function for a plotting callback which can be used by the reporter
# function of an embedding to plot the current (two-dimensional) embedding.
#
# @param x Data frame.
# @param colors Vector of colors for \code{x}.
# @param color_name Name of a column in \code{x} that contains colors to use.
# @param labels Vector of labels for \code{x}. Ignored if \code{colors} or
#   \code{color_name} is provided.
# @param label_name Name of the label column in \code{x}. Ignored if
#  \code{colors}, \code{color_name} or \code{labels} is provided.
# @param label_fn Function with the signature \code{label_fn(labels)} where
# \code{labels} is a vector of labels for each point in the data set. The
# function should return a vector of labels suitable for displaying in the
# plot.
# @param color_scheme Either a color ramp function or the name of an
#   RColorBrewer color scheme to use. The latter requires the RColorBrewer
#   package to be installed and loaded.
# @param mat_name The name of the matrix containing the embedded data in the
# output list \code{out} which will be passed to the plot function.
# @param cex Numeric \strong{c}haracter \strong{ex}pansion factor;
#   multiplied by \code{\link[graphics]{par}("cex")} yields the final
#   character size of the labels.
# @return Function which will take an output list, and produce a 2D plot of
# the embedding.
# @seealso \code{make_reporter} for how to use this function for
# configuring visualization of the progress of an embedding.
# @examples
# # For s1k dataset, plot 2D embedding with "Label" factor to identify each
# # point on the plot
# make_plot(s1k, labels = iris$Label)
#
# # For iris dataset, plot 2D embedding with first two characters of the
# # "Species" factor to identify each point on the plot
# make_plot(iris, labels = iris$Species, label_fn = make_label(2))
#
# # Should be passed to the plot_fn argument of the reporter factory function:
# \dontrun{
#  embed_prob(reporter =
#               make_reporter(report_every = 100,
#                             normalize_cost = TRUE,
#                             plot =
#                               make_plot(iris, labels = iris$Species)),
#                                     ...)
# }
# @family sneer plot functions
make_plot <- function(x,
                      colors = NULL,
                      labels = NULL,
                      label_fn = function(labels) {
                        labels
                      },
                      color_scheme = grDevices::rainbow,
                      cex = 1,
                      show_labels = FALSE,
                      equal_axes = FALSE,
                      mat_name = "ym") {

  embedding_plot <- make_embedding_plot(x,
                                        colors = colors,
                                        labels = labels,
                                        label_fn = label_fn,
                                        color_scheme = color_scheme,
                                        cex = cex,
                                        show_labels = show_labels,
                                        equal_axes = equal_axes)
  function(out) {
    embedding_plot(out[[mat_name]])
  }
}

# Embedding Plot Using \code{graphics} Library
#
# Create a function which when invoked on a 2D matrix, plots the embedding
# with color-coded labels.
#
# x Data frame.
# colors Vector of colors for x. If not supplied, but the labels parameter
#   does have a value, then the labels will be used to map to a vector of
#   colors.
# labels Factor vector containing one label or category per point in. If the
#   text of these labels will be displayed instead of points, so make sure
#   your dataset is small and the labels are short (preferably both).
# label_fn Function to generated modified (probably shorter) labels.
#   It should take one argument - the vector of labels, and return a vector of
#   labels that will be used for display. Ignored if the labels parameter is
#   not used.
# color_scheme The color scheme to map the labels provided by labels or
#   label_name to colors. Either a color ramp function or the name of an
#   RColorBrewer color scheme to use. The latter requires the RColorBrewer
#   package to be installed and loaded. Ignored if the labels parameter is
#   not used.
# cex The size of the points or text (if labels or label_name is supplied).
#   Has the usual meaning when used with the graphics::plot command.
# show_labels if TRUE, then if labels are provided, plot them (or the output
#   of label_fn if that is non-NULL) instead of points.
# equal_axes if TRUE, then the range of x and y axes will be the same.
#
# Returns a function which takes a matrix of 2D coordinates and produces a 2D
# plot of the embedding.
make_embedding_plot <- function(x,
                                colors = NULL,
                                labels = NULL,
                                label_fn = NULL,
                                color_scheme = grDevices::rainbow,
                                cex = 1,
                                show_labels = FALSE,
                                equal_axes = FALSE) {
  # If labels were provided but no colors, let's map labels to colors now
  if (!is.null(labels)) {
    if (is.null(colors)) {
      colors <- factor_to_colors(x = labels, color_scheme = color_scheme)
    }
    if (show_labels) {
      if (!is.null(label_fn)) {
        text <- label_fn(labels)
      }
      else {
        text <- labels
      }
    }
    else {
      text <- NULL
    }
  }

  function(ym) {
    embed_plot(ym, colors = colors, cex = cex, text = text,
               equal_axes = equal_axes)
  }
}

# Labels for 2D Embedding Plot
#
# Even quite short labels can create a crowded looking embedding plot.
# Use this function to only print the first few characters of each label.
#
# @param num_label_chars The number of characters to plot from the label
# for each data point.
# @return a function which can be passed to the \code{label_fn} parameter
# of the \code{make_plot} function.
make_label <- function(num_label_chars = 1) {
  partial(substr, start = 0, stop = num_label_chars)
}

# Embedding Plot Using \code{ggplot2} and \code{RColorBrewer} Library
#
# Creates a function which can be used to visualize embeddings from sneer
# output results for a particular dataset using ggplot2.
#
# @note Use of this function requires that the \code{ggplot2} and
# \code{RColorBrewer} packages be installed.
#
# @param df Data frame containing label information for the embedded data.
# @param label_name Name of the label column in \code{df}. Ignored if
#  \code{labels} is provided.
# @param labels Vector of labels.
# @param mat_name The name of the matrix containing the embedded data in the
#   output list \code{out} which will be passed to the plot function.
# @param size Size of the points.
# @param color_scheme Color ramp function or string giving the name of a
#  ColorBrewer scheme. To see the
#  available ColorBrewer schemes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following palettes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @return Function with signature \code{plot_fn(out)} where \code{out} is
# a return value from a sneer embedding function. On invocation, the
# data will be plotted.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
#
# @examples
# \dontrun{
# mds_iris <- embed_dist(iris[, 1:4])
# iris_view <- make_qplot(iris, "Species")
# iris_view(mds_iris)
# }
make_qplot <- function(df, label_name = "Label", labels = NULL, mat_name = "ym",
                       size = 1,
                       color_scheme = "Set1",
                       legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("make_qplot function requires 'ggplot2' package")
  }
  if (class(color_scheme) == "character" &&
      !requireNamespace("RColorBrewer", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("make_qplot function using ColorBrewer names requires ",
         "'RColorBrewer' package")
  }
  embedding_plot <- make_embedding_qplot(df, label_name, labels = labels,
                                         size = size,
                                         color_scheme = color_scheme,
                                         legend = legend,
                                         legend_rows = legend_rows)

  function(out) {
      embedding_plot(out[[mat_name]])
  }
}

# Embedded Coordinates Plot Using \code{ggplot2} Library
#
# Creates a ggplot2 function for embeddings of a given dataset with coloring
# from RColorBrewer color schemes or a color ramp function.
#
# @note Use of this function requires the \code{ggplot2} package be installed.
# To use a ColorBrewer color scheme name, the \code{RColorBrewer} package must
# be installed.
#
# @param df Data frame containing label information for the embedded data.
# @param label_name Name of the label column in \code{df}.
# @param size Size of the points.
# @param color_scheme Color ramp function or a string giving the name of a
# ColorBrewer color scheme To see
#  the available schemes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following color schemes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @return Function with signature \code{plot_fn(ym)} where \code{ym} is a
# 2D matrix of embedded coordinates of data set \code{x}. On invocation, the
# data will be plotted.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
#
# @examples
# \dontrun{
# # make two different embeddings of the iris dataset
# prcomp_iris <- prcomp(iris[, 1:4], scale. = TRUE, retx = TRUE)
# mds_iris <- embed_dist(iris[, 1:4], method = mmds(eps = 1e-4),
#                        opt = bold_nag(),
#                        init_out = out_from_matrix(prcomp_iris$x[, 1:2]),
#                        max_iter = 40)
# iris_view <- make_embedding_qplot(iris, "Species")
# iris_view(prcomp_iris$x)
# iris_view(mds_iris$ym)
# }
make_embedding_qplot <- function(df, label_name = "Label", labels = NULL,
                                 size = 1,
                                 color_scheme = "Set1",
                                 legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("make_embedding_qplot function requires 'ggplot2' package")
  }
  if (class(color_scheme) == "character" &&
      !requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("make_embedding_qplot function requires 'RColorBrewer' package")
  }

  function(ym) {
    colnames(ym) <- NULL
    scatterqplot(df, x = ym[, 1], y = ym[, 2], label_name = label_name,
                 labels = labels,
                 size = size, color_scheme = color_scheme,
                 x_label = "D1", y_label = "D2",
                 legend = legend, legend_rows = legend_rows)
  }
}

# Interpolated ColorBrewer Palette
#
# Returns a vector of colors from the specified palette, interpolated if the
# number of requested colors is larger than the number of colors in the
# palette. Sequential and Diverging palettes are suitable for numerical scales.
# The Qualitiative palettes are intended for categorical values.
#
# Sequential palettes names:
#  Blues BuGn BuPu GnBu Greens Greys Oranges OrRd PuBu PuBuGn PuRd Purples
#  RdPu Reds YlGn YlGnBu YlOrBr YlOrRd
# Diverging palette names:
#  BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
# Qualitative:
#  Accent Dark2 Paired Pastel1 Pastel2	Set1 Set2	Set3
# @param name Name of the palette.
# @param ncolors Number of colors desired.
# @return Vector of \code{n} colors from the palette.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
color_brewer_palette <- function(name, ncolors) {
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("color_brewer_palette function requires 'RColorBrewer' package")
  }
  make_color_brewer_ramp(name)(ncolors)
}

# Interpolated ColorBrewer Ramp
#
# Creates a color ramp function using the ColorBrewer palettes, with
# interpolation if the requested number of colors exceeds the maximum number of
# colors in the palette. Sequential and Diverging palettes are suitable for
# numerical scales. The Qualitiative palettes are intended for categorical
# values.
#
# Sequential palettes names:
#  Blues BuGn BuPu GnBu Greens Greys Oranges OrRd PuBu PuBuGn PuRd Purples
#  RdPu Reds YlGn YlGnBu YlOrBr YlOrRd
# Diverging palette names:
#  BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
# Qualitative:
#  Accent Dark2 Paired Pastel1 Pastel2	Set1 Set2	Set3
#
# @param name Name of the palette.
# @return Function accepting an integer n as an argument and returning n colors.
#
# @note This function requires the RColorBrewer package to be installed and
#  loaded.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
make_color_brewer_ramp <- function(name) {
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("colorBrewerPalette function requires 'RColorBrewer' package")
  }
  if (!name %in% rownames(RColorBrewer::brewer.pal.info)) {
    stop("Unknown ColorBrewer name '", name, "', must be one of ",
         paste(rownames(RColorBrewer::brewer.pal.info), collapse = ", "))
  }

  function(n) {
    max_colors <- RColorBrewer::brewer.pal.info[name,]$maxcolors
    n <- max(n, 3)
    if (n <= max_colors) {
      RColorBrewer::brewer.pal(n, name)
    }
    else {
      grDevices::colorRampPalette(RColorBrewer::brewer.pal(max_colors, name))(n)
    }
  }
}
