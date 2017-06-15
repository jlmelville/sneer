---
title: "Visualization"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Previous: [Analysis](analysis.html). Next: [References](references.html). Up: [Index](index.html).

During the embedding, `sneer` will display the current state of the output
configuration. But you can also use a standalone function to visualize the
coordinates (or any 2D matrix), so we'll look at that first while we run
through the different options available.

## `embed_plot`

`embed_plot` is a function you can use to easily view the results of the plot.
Just looking at the output as an undifferentiated mass is rarely enlightening,
so you will normally want to color the points in some way. Three main strategies
come to mind:

* If you have already got a column in the data frame that contains colors,
use that.
* Where the data falls into a set of categories, map from those categories
to a color.
* If you have a number associated with each point (e.g. from one of vectors
you can [export](exported-data.html) from the embedding), you might want
to map those numbers to a color scale and project the values onto the points.

`embed_plot` can help you with that. However, if you go with the 
categories-to-colors plot, you won't get a legend, so you will be able to 
discern any clustering by the colors, but you won't know which color maps
to which category. Sorry. I was unable to find a satisfactory solution with
the standard `graphics::plot` commands that could guarantee the legend would
be in a sensible place and location for arbitrary numbers of categories and
category names of arbitrary length. 

Anyway, let's run through some examples. Let's do a quick tsne embedding on
our old friend the iris dataset, and lets extract the degree centrality of
each node too so we have something numeric to project onto the plot.

```R
tsne_iris <- sneer(iris, max_iter = 200, ret = c("deg"))
```

### Default

Let's take a look at the embedding:

![`embed_plot(tsne_iris$coords)`](embed-plot-default.png)

Well, that's garish. This is because you haven't provided any color information,
so the default is that each point gets its own color from the default color 
scheme, which happens to be the `grDevices::rainbow` function. This might be 
useful for some simulation data sets where the data is produced by a sequential 
function of some kind (e.g. a random walk or a path around a circle or some
other parametric function) and the progression of the color gives some clue
about location.

### Explicit `colors`

If you just want everything to be one color, pass that to the `colors` 
parameter:

![`embed_plot(tsne_iris$coords, colors = "black")`](embed-plot-all-black.png)

If you have a preset vector of colors, that makes a lot more sense. Perhaps, 
inspired by the technicolor dream plot of the first example, you want each iris 
to have its own random color:

```R
niris <- nrow(iris)
iris_colors <- rgb(runif(niris), runif(niris), runif(niris))
```

![`embed_plot(tsne_iris$coords, colors = iris_colors)`](embed-plot-random-colors.png)

Beautiful.

More sensibly, the datasets used in 
[How to use t-SNE Effectively](http://distill.pub/2016/misread-tsne/), and
available in the github package [snedata](https://github.com/jlmelville/snedata)
have a pre-calculated `color` column.

Do the installation, and then carry out PCA on the three clusters simulation
data:

```R
devtools::install_github("jlmelville/snedata")
library("snedata")
three_c <- three_clusters_data(50)
pca_3c <- sneer(three_c, method = "pca")
```

View the results with a less foolish set of colors than random:

![`embed_plot(pca_3d$coords, colors = three_c$color)`](embed-plot-3c.png)

### Mapping from factor levels to colors

Back to our long-suffering iris example. It consists of measurements of the
sepal and petal dimensions of 150 flowers, from 3 species: *Iris setosa*,
*Iris versicolor* and *Iris virginica*. That species information is in the
`iris$Species` column, so it would be natural to map from those categories
to colors.

To do this, pass the `iris$Species` column to the `x` argument. This argument
can be a data frame or a column. `embed_plot` will try and work out what to
do based on what it's passed. In this case, the result is:

![`embed_plot(tsne_iris$coords, x = iris$Species)`](embed-plot-iris-species.png)

Three species, three colors (sorry you can't see a legend).

### `color_scheme`

If you don't like the color scheme used, you can change it by passing a different
color function to `color_scheme`. For example, here's the use of the 
`topo.colors` function, also from the `grDevices` package, like `rainbow`:

![`embed_plot(tsne_iris$coords, x = iris$Species, color_scheme = topo.colors`)](embed-plot-iris-topo.png)

You can write your own color function as long as it follows the same interface
as the ones in `grDevices`: you pass in a single integer, `n`, and you get back
a vector of `n` colors. For instance, perhaps you really liked those random
colors. Enshrine your non-deterministic creativity as a function:

```R
random_colors <- function(n) {
  rgb(runif(n), runif(n), runif(n)
}
```

Now, whenever you want each iris species to get a random color, it's as simple
as:

![`embed_plot(tsne_iris$coords, x = iris$Species, color_scheme = random_colors`)](embed-plot-iris-random-series.png)

### `RColorBrewer`

If, like me, you have little creativity when it comes to picking color schemes,
I recommend just deferring to the [ColorBrewer](http://colorbrewer2.org/)
color schemes (although [other color schemes exist](https://tradeblotter.wordpress.com/2013/02/28/the-paul-tol-21-color-salute/)).
These are available both for numerical scales, and for mapping categorically.
Use the [RColorBrewer](https://cran.r-project.org/package=RColorBrewer) 
package to get access to them in R. You need to install and load it 
manually if you want to use it with `embed_plot`:

```R
install.packages("RColorBrewer")
library(RColorBrewer)
```

To use a scheme from ColorBrewer, pass the *name* of a scheme to the 
`color_scheme` parameter, rather than a function. Find the names by running
`RColorBrewer::brewer.pal.info()` or `RColorBrewer::display.brewer.all()`.

For mapping factors to colors, you should be using one of the "qual" (short for
"qualitative") schemes, for example "Accent":

![`embed_plot(tsne_iris$coords, x = iris$Species, color_scheme = "Accent")`](embed-plot-accent.png)

If the data set you are using has more factors than the color scheme has colors,
`embed_plot` attempts to interpolate the colors into a new set of colors. Not
ideal, but better than nothing.

### Passing a data frame to `x`

Previously, I mentioned that if you pass a data frame rather than a vector to
the `x` parameter, the function will attempt to do the right thing. What this
means in practice is:

* it first of all looks for a column of strings, where each member can be
interpreted as a color. If it finds one, it will use that for colors. If it 
finds more than one, it uses the last column it finds.
* otherwise, if looks for a factor column. If it finds one, it maps that to
colors as if you had passed the column directly. Again, if more than one
column is found, it uses the last one it finds.
* otherwise, as you have seen, you get one color per point.

This is a convenience for when you have multiple data sets you're looking at,
they have different column names and types that you want to use for coloring
and you don't want to remember them all.

As `x` is the second argument to `embed_plot` positionally, you don't even
need to refer to it by name, which makes using `embed_plot` even easier. For 
example, the following pairs of invocations are equivalent to each other: 

```R
embed_plot(pca_3d$coords, colors = three_c$color)
embed_plot(pca_3d$coords, three_c) # will find the "color" column

embed_plot(tsne_iris$coords, x = iris$Species)
embed_plot(tsne_iris$coords, iris) # will find the "Species" column

embed_plot(tsne_iris$coords, colors = iris_colors) 
# detects it's already been passed a color vector when passed as 'x' argument
embed_plot(tsne_iris$coords, iris_colors) 
```

### Factor levels as text

In some cases, you may want to plot the factor levels you're using for the color
scheme as labels on the plot, instead of the points themselves, especially 
because of the lack of a legend. To do this, set the `text` argument to the 
column you want to use. Let's finally find out which species is which in the
iris plot:

![`embed_plot(tsne_iris$coords, iris, text = iris$Species)`](embed-plot-iris-text.png)

Um, I suppose you can just about make out the names. Perhaps it'll look better
if we reduce the size of the text with the `cex` parameter and only plot
the first two characters of the species name?

![`embed_plot(tsne_iris$coords, iris, text = substr(iris$Species, 1, 2), cex = 0.7)`](embed-plot-iris-text2.png)

That's a bit better. As you can see, this probably isn't a great option, unless
you have a small dataset with short labels.

### Mapping numeric vectors

You can also map a numeric vector to a point. Choosing a good color scheme is
very important here, so I recommend the ColorBrewer sequential and diverging 
schemes. Here's the iris plot, with the degree centrality - a measure of how 
"important" each point is - projected onto the plot, with the "Blues" 
sequential color scheme:

![`embed_plot(tsne_iris$coords, tsne_iris$deg, color_scheme = "Blues")`](embed-plot-deg-blues.png)

The dark blue points are inside the clusters, with the least important points
on the edges. Makes sense. Good job, t-SNE (the degree centrality is calculated
using the input probability only so it's not using the output data directly).

As you can see, sequential schemes highlight large values. If you want to 
highlight values at both ends of the scale, use a diverging scale, like 
`RdYlGn`:

![`embed_plot(tsne_iris$coords, tsne_iris$deg, color_scheme = "RdYlGn")`](embed-plot-deg-rdylgn.png)

### Extra parameters for numerical mapping

There are some extra parameters that can be used to control the display of the
numerical mapping. Sometimes, you may only want to see the points with the 
largest values. Use the `top` argument for this. For example, here are the 
top 15 most important iris points:

![`embed_plot(tsne_iris$coords, tsne_iris$deg, color_scheme = "Blues", top = 15)`](embed-plot-deg-top15.png)

By default the color scheme is mapped to the extent of the numerical vector
you provide. Sometimes, there's an absolute scale you may want to use, however.
For instance, the neighborhood preservation value, discussed in greater detail
in the [Analysis](analysis.html) section takes a value between 0 and 1, 
representing an embedded point that has retained none or all of its neighbors,
respectively.

Let's see how well PCA does at preserving each point's 25 neighbors. Note
that we need to `ret`urn the input distance matrix (`dx`) and output distance
matrix (`dy`) for the `nbr_pres` calcuation:

```R
pca_iris <- sneer(iris, ret = c("dx", "dy"), method = "pca")
pca_nbrp25 <- nbr_pres(pca_iris$dx, pca_iris$dy, 25)
```

Running `summary(pca_nbrp25)` shows that PCA does a pretty good job:

```
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 0.5600  0.8800  0.9200  0.8939  0.9600  1.0000 
```

If we plot the nbr preservation values as colors (with the size of the points
increased via the `cex` parameter):

![`embed_plot(pca_iris$coords, pca_nbrp25, color_scheme = "Spectral", cex = 1.5)`](embed-plot-nbr.png)

There are a few red points in the middle of the two right hand clusters - that's
where the points have the lowest neighborhood preservations are. But given 
they're never lower than 0.56, you could replot this, using the full preservation
scale, by passing the lower and upper limits to the `limits` argument:

![`embed_plot(pca_iris$coords, pca_nbrp25, color_scheme = "Spectral", cex = 1.5, limits = c(0, 1))`](embed-plot-nbr-limits.png)

Doesn't seem so bad now those red points have turned yellow.

### Miscellaneous

There are a few other commands that can modify the look of the plot.

As we've seen in a few places, `cex` scales the size of the points (or the text 
if the `text` argument is used). Set it to less than 1 to make the points
smaller than default, and greater than 1 to make them bigger.

`equal_axes`, if set to `TRUE` will make the extent of the X and Y axis the 
same. This is useful if you would rather have a 1:1 aspect ratio, at the
expense of more whitespace and in your plot. Here's the iris plot yet again,
now with equal X and Y axes:

![`embed_plot(tsne_iris$coords, iris, equal_axes = TRUE)`](embed-plot-equal-axes.png)

The clusters now look a lot less "tall" and stretched in the vertical 
direction.

## Plotting in `sneer`

Much of what has been described here applies to the plots which appear during
the embedding process when running `sneer`. Here are the options to be aware
of.

### `colors`, `color_name`

You may provide a vector of colors directly to `sneer` via the `colors` 
parameter, or the name of a column in the data frame via `color_name`.

```R
sneer(three_c, colors = three_c$color)
sneer(three_c, color_name = "color") # the same as the previous command
```

### `labels`, `label_name`

Similarly, you can map from a factor to colors with the `labels` and 
`label_name` parameters:

```R
sneer(iris, labels = iris$Species)
sneer(iris, label_name = "Species") # the same as the previous command
```

### Defaults

Just like `embed_plot`, `sneer` will try and find a color of label column from
the data frame you pass it:

```R
sneer(iris, labels = iris$Species)
sneer(iris) # equivalent to the above
```

The same rules for finding columns applies for `sneer` as for `embed_plot`:
color columns are returned in preference to factor columns, and if nothing is
found, one color is used per point. If more than one column is found, the
last column is used.

### `color_scheme`

You can specify the color scheme just like in `embed_plot`: pass either a color
function like `rainbow` or the *name* of a ColorBrewer color scheme, e.g. 
`"Set1"`.

### `plot_type`

If you embed in other than 2 dimensions, you won't see a plot. You can also
turn off plotting by setting `plot_type = "none"`.

There is also support for using `ggplot2`. As usual, you need to install and
load it yourself:

```R
install.package("ggplot2")
library(ggplot2)
```

Then, set `plot_type = "ggplot2"` (or just `"g"`):

![`tsne_iris <- sneer(iris, plot_type = "ggplot2")`](tsne-iris-ggplot2.png)

Could that be... yes it is. It's a legend at long, long last. This is the big
advantage of using the `ggplot2` plotting during the embedding. It's not
currently supported by `embed_plot`.

Before you get too excited about legends and everything there are some downsides
to using it. First, even though positioning the legend works much better with
`ggplot2` than using `plot`, you still can have a horrible things happen if
you have a very large legend (because of lots of categories or long category 
names). You can try fiddling with the `legend_rows` parameter in this case,
which specifies how many rows to try and lay the legend out on. This might help
prevent the legend from getting too high or wide. If the worst comes to the 
worst you can set the `legend` parameter to `FALSE` to turn it off.

Also, coloring the plot has to be done by mapping from a factor to a
color scheme currently (i.e. no columns of explicit color names). Finally,
you can't choose to display factor labels instead of points (but at least you
have a legend).

### `cex`

As with `embed_plot`, the `cex` parameter changes the size of the plotted 
points.

### `equal_axes`

Set this to `TRUE` to have the axes have the same scale.

### `plot_labels`

If `TRUE`, whatever factor column provided to `sneer` (or it finds itself)
will be plotted as labels on the embedding plot. As discussed under the 
`text` parameter of `embed_plot`, has limited (but not zero) use.

## `embed_plotly`

As an alternative to `embed_plot`, you can try out `embed_plotly`, which 
uses the JavaScript charting library [plotly](https://plot.ly) and its R API
to generate embedding plots. You will need to install the 
[plotly](https://cran.r-project.org/package=plotly) package, ensuring the
version you are using is at least version 4:

```R
install.package("plotly")
library("plotly")
packageVersion("plotly") # This needs to begin with '4.'
```

If you're using [RStudio](https://www.rstudio.com/), calling `embed_plotly`
won't seem all that different to using `embed_plot`. However, if you're using
the usual R shell, then it will open the plot in your default web browser. If
you can handle that, then there are a few advantages to using plotly.
First:

![`embed_plotly(tsne_iris$coords, iris$Species, color_scheme = "Accent")`](embed-plotly-categorical.png)

Yep, you get a legend.

Second:

![`embed_plotly(tsne_iris$coords, tsne_iris$deg, color_scheme = "Blues")`](embed-plotly-numeric.png)

And numeric vectors gets a color scale. Also, if you pass in some labels as
the `text` argument, they will show up in a tooltip if you hover over a point.

Third, there are lots of panning and zooming options if you want to 
interactively explore different areas of the plot in more detail.

The only minor downside to the use of `embed_plotly` is that it doesn't support
the `limits` and `top` arguments when used with a numeric vector. A small
price to pay.

Previous: [Analysis](analysis.html). Next: [References](references.html). Up: [Index](index.html).
