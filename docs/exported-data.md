---
title: "Exported Data"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Previous: [Reporting](reporting.html). Next:[Analysis](analysis.html). Up: [Index](index.html).

## `sneer` return value

Having run an embedding: 

```R
iris_tsne <- sneer(iris)
```

what exactly does `iris_tsne` give you? It's a list with something like the
following:

* `coords` - a matrix of all-important coordinates. It's dimensions are `n` x 
 `ndim` where `n` is the number of observations that were embedded and `ndim`
 is the number of output dimensions. That will normally be 2.
* `method` - a string containing the embedding method name.
* `cost` - the final cost associated with the output configuration in `coords`.
  See the [Reporting](reporting.html) section for more on this.
* `norm_cost` - the normalized version of `cost`. Again, see the 
  [Reporting](reporting.html) section for more.
* `iter` - the iteration number at which the embedding stopped.

## `ret`

You can also ask for some other data. Some of it might be useful for 
diagnostics, debugging, visualization or as input to another algorithm. To 
get access to it, pass a vector of names to the `ret` parameter.

The names you can ask for are:

* `pcost` - The final `cost`, decomposed into the sum of *n* values, with *n* 
  being the number of points. Be aware that these individual components don't 
  have to be positive, e.g. if the cost function is a divergence. For the `pca`
  method, the `mmds` cost function is used.
* `x` - input coordinates after [Preprocessing](preprocessing.html) and column
  filtering.
* `dx` - input distance matrix. Calculated if not present. Note that this of 
  class `matrix`, not `dist`.
* `dy` - output distance matrix. Calculated if not present. Note that this of 
  class `matrix`, not `dist`.
* `p` - input probability matrix. Only available if using a probability-based
  embedding.
* `q` - output probability matrix. Only available if using a probability-based
  embedding.
* `prec` - vector of input kernel precisions. Corresponds to the 
  values summarized as `prec` in the console log after input initialization. 
  Only available if using a probability-based embedding. See the 
  [Input Initialization](input-initialization.html) section for more on the
  exact definition of the precision.
* `dim` - vector of intrinsic dimensionalities for each observation, calculated 
  according to the method given in the 
  [multiscale JSE](http://dx.doi.org/10.1016/j.neucom.2014.12.095) paper.
  These are meaningless if not using the default exponential 
  \code{perp_kernel_fun}. See the 
  [Input Initialization](input-initialization.html) section for more.
* `deg` Vector of degree centrality of the input observations as used in 
  [ws-SNE](http://jmlr.org/proceedings/papers/v32/yange14.html). 
  Calculated if not present. A summary of this vector appears in the console 
  log after input initialization if carrying out ws-SNE (by setting 
  `method = "wssne"`).
* `dyn` Dynamically optimized non-coordinate parameters. A list containing 
  sublists, with names corresponding to the names of the parameters optimized. 
  This is avaiable if using 
  [Inhomogeneous t-SNE](http://dx.doi.org/10.1007/978-3-319-46675-0_14)
  (`method = "itsne"`), where `dyn` will contain a list `dof`, which contains
  the optimized degrees of freedom parameter for each output coordinate.
  If using `method = "dhssne"` method, dyn contains a list called `alpha`,
  which contains the global alpha optimized alpha value. See the 
  [Embedding Methods](embedding-methods.html) for more on DHSSNE.
* `costs` All the costs which were logged to screen during the optimization, as
  a matrix, with the iteration number in the first column.

For every name you provide, the list returned by `sneer` will contain an extra
item in the list with the same name. If you ask for an item that doesn't make
sense, e.g. a probability matrix from an embedding method that never calculated
one, the request is silently ignored. Otherwise, the value will be calculated
if possible, e.g. even though degree centrality is only used in ws-SNE, you can
ask for it to be returned from any probability-based embedding, because it's
related to the input probabilities, so it can be calculated.

```R
# return the point-wise cost function, output distance matrix and the degree centrality
# in tsne_iris$dy and tsne_iris$deg, respectively
tsne_iris <- sneer(iris, ret = c("pcost", "dy", "deg"))

# returns the output distance matrix in sammon_iris$dy, but no
# sammon_iris$deg, because that would require the presence of data that
# Sammon Mapping doesn't create
sammon_iris <- sneer(iris, ret = c("dy", "deg"))
```

Although `sneer` requires input distance matrix data to be of class `dist`, 
internally, it uses objects of class `matrix` for all matrices, even  distance 
matrices. So `dy` and `dx` will be of class `matrix`. If you want to pass `dy` 
or `dx` back into `sneer` for some crazy scheme or other, you will need to cast 
them into `dist` objects via `as.dist`:

```R
s1k_tsne <- sneer(s1k, ret = c("dy"))
# ... work some unseen magic on s1k_tsne$dy ...
# return it to sneer for more processing
s1k_mmds <- sneer(as.dist(s1k_tsne$dy), method = "mmds", labels = s1k$Label)
```

You may find some use for the numerical vectors that can be returned with `ret`
by coloring the embedding plot based on their magnitude. The
[Visualization](visualization.html) section can tell you more.

Previous: [Reporting](reporting.html). Next:[Analysis](analysis.html). Up: [Index](index.html).
