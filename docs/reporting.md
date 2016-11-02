---
title: "Reporting"
output: html_document
---
Previous: [Embedding Methods](embedding-methods.html). Next:[Exported Data](exported-data.html). Up: [Index](index.html).

One of the most useful things `sneer` does during optimization is provide
a plot of the current state of the output configuration. Details of that
gets its own section: [Visualization](visualization.html). This section
covers both the textual part of the logging, its frequency and how it knows
when to stop.

During optimization, various information is logged to the screen. Some of it
appears during input initialization (see the 
[Input Initialization](input-initialization.html) section). We'll cover some of
the rest here.

The optimization logging messages look like:

```Iteration #50 cost = 0.503 norm = 0.3987 rtol = 20```

The `Iteration` value is the iteration number. Nothing surprising there. 

The `cost` is the value of the cost function that's specific to the embedding
method. For example, for t-SNE, the `cost` is the Kullback-Leibler divergence. 
For Sammon Mapping, the cost is the Sammon Stress. It's not particularly
interesting what the number is, as long as you see the number going down.
For a given parameterization, the smaller the number, the better. However, 
comparing costs between different methods or different data sets is not 
meaningful.

It's also a bit hard to tell on an absolute scale how well you're doing. A 
perfect embedding will give a cost of 0, but is `cost = 0.503` good or bad? 
The `norm` value attempts to provide a normalization that helps. `0` is still
perfect, but a `norm` of `1` is intended to represent the value you'd get if
you completely ignored all the input data you were given and made all the 
distances (or probabilities) equal, which you can achieve by setting all
the distances to zero. Hence, if you initialize the output coordinates
with the small gaussian distribution (`init = "r"` and see 
[Output Initialization](output-initialization.html) for more details), your
initial `norm` cost will be very close to 1 for most embedding methods.

If you use Sammon Mapping (`method = "sammon"`), the `cost` used is the Sammon
Stress, which is already normalized, so no `norm` value is provided.

Additionally, for PCA (`method = "pca"`), metric MDS (`method = "mmds"`) and
Sammon Mapping, a `kruskal_stress` value is provided. This reports Kruskal
Stress formula 1, which is related to the square loss of the input and output
distances. It's not reported for other embedding methods which aren't concerned
with preserving distances.

### `tol`

The `rtol` value in the output measures the relative tolerance between the 
currently reported cost value and that of the previous report. As mentioned in 
the section on [Optimization](optimization.html), the embedding will stop if the
`rtol` value falls below the argument provided to the `tol` parameter.

```R
tsne_iris <- sneer(iris, tol = 0.01)
```

### `report_every`

You can change how often optimization progress is logged to screen by changing
the `report_every` value. For example, if you set it `100`, it will report
every 100 iterations.

Watch out if you set the value low. First, if you are plotting the embedding, 
I've had the R session in RStudio crash on me with a complaint about graphics.
This may be something to do with trying to get a graphics plot to refresh too
quickly.

Second, because the tolerance is checked after every logged cost value, if you
report too often and set `tol` too high, the optimization will converge because
the cost won't have changed enough between reports. The best course of action
is to set `tol = 0` under these conditions.

```R
# If you like a talkative embedding
tsne_iris <- sneer(iris, report_every = 1, tol = 0)
```

The other way to affect how many times the information is logged to console
is to change the `max_iter` argument that determines the total number of
iterations. Also see the [Optimization](optimization.html) section for more
on that.

Previous: [Embedding Methods](embedding-methods.html). Next:[Exported Data](exported-data.html). Up: [Index](index.html).
