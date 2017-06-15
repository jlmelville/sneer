---
title: "Optimization"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Previous: [Output Initialization](output-initialization.html). Next: [Embedding Methods](embedding-methods.html). Up: [Index](index.html).

I have exposed a lot of options to do with optimization, but I don't really
recommend using most of them, except out of morbid curiosity. Here are the 
ones that matter:

## `max_iter`

This controls the maximum number of iterations. The default is 1000, which
is probably way too high, but it depends on your data set.

Often, the data is in a pretty good shape after 100 iterations, after which
it's just fine tuning, with various clusters very slowly moving away from 
each other. 

For example, let's run t-SNE on the `iris` data set for 100 iterations:

![`iris_tsne <- sneer(iris, max_iter = 100)`](iris-tsne-100.png)

And then do it again, but this time let it go for 1000 iterations:

![`iris_tsne <- sneer(iris, max_iter = 1000)`](iris-tsne-1000.png)

Yes, they look different. But not that different that you would draw 
qualitatively the wrong conclusions.

If you attempt to quantify the result of your embedding, for example by using
some of the methods mentioned in the [Analysis](analysis.html) section, these
can be sensitive to the coordinates. So if you're comparing different methods, 
ensure you are not unfairly maligning a method because you didn't let it 
converge. But most of the time, embedding methods are used to eyeball a data 
set, not for quantitative dimensionality reduction, so you probably don't care. 
Additionally, a lower error embedding can sometimes look *worse* than a 
configuration from earlier iterations. The overall structure of the embedding, 
e.g. the number of relative positons of clusters, appears quite early in the 
optimization. These clusters then tend to spend the rest of the time increasing 
their relative distances and growing tighter. You then have to spend more time 
panning and zooming than you would otherwise. So don't be afraid to crank down
`max_iter`.

## `tol`

Another way to stop the embedding early is to adjust the tolerance parameter,
`tol`. If you jump ahead to the [Reporting](reporting.html), you'll see a
discussion of the messages logged to the console during the optimization phase.
What you should know is that if the value of `rtol` reported in the output
ever falls below `tol`, then optimization is considered converged. The
default is `1e-5`, which once again errs on the side of caution. It pretty much
ensures that if the formatted `cost` or `norm` values in the output log are still
changing, it won't stop early. But once again, you might want to turn that way
down for simple visualizations. A `tol` of `0.01` is pretty reasonable:

```R
iris_tsne <- sneer(iris, tol = 0.01) # stops after 200 steps
```

## `opt`

Most t-SNE implementations follow the optimization technique given by the 
[t-SNE paper](http://jmlr.org/papers/v9/vandermaaten08a.html): the direction
of optimization is basic gradient descent with a momentum term, and an 
adaptive step size, which requires setting an initial learning rate, `epsilon`.

This works well for t-SNE, and it's fast, but in my experience it can 
cause divergence when a non t-SNE embedding method is used. For this reason, 
it's not the default. If you want to use it, set the `opt` parameter to 
`"tsne"` and the `epsilon` value for the learning rate:

```R
iris_tsne <- sneer(iris, opt = "tsne", epsilon = 500, scale_type = "m")
```

This form of optimization is usually combined with early exaggeration and 
random initialization. See the [Input Initialization](input-initialization.html)
and [Output Initialization](output-initialization.html) sections for more 
details, but for an authentic-ish t-SNE experience, run:

```R
iris_tsne <- sneer(iris, opt = "tsne", epsilon = 100, exaggerate = 4,
                  exaggerate_off_iter = 50, perplexity = 30, init = "r")
```

The default optimizer uses the limited-memory Broyden-Fletcher-Goldfarb-Shanno
(L-BFGS) optimizer and is used in the 
[JSE paper](http://dx.doi.org/10.1016/j.neucom.2012.12.036). 

```R
s1k_tsne <- sneer(s1k, opt = "L-BFGS")
```

If you want to try using the conjugate gradient optimization method (which is 
the method used in the [NeRV](http://www.jmlr.org/papers/v11/venna10a.html) 
paper, use:

```R
s1k_tsne <- sneer(s1k, opt = "CG") 
```
In case you are curious, the specific flavor of CG used is the Polak-Ribiere
update with restart (sometimes called 'PR+').

The [Spectral Directions](https://arxiv.org/abs/1206.4646) optimizer is 
specially crafted for certain neighbor embedding functions, including t-SNE.
Specifically, using the jargon referenced in the [gradients](gradients.html)
theory page, only methods which use the "pairwise" normalization method work
with it.

The Spectral Directions method relies on sparsity for it to be performant, 
which is something `sneer` doesn't currently support. You may therefore run 
into memory problems if you use it with large data sets, but you're going to 
run into memory problems with large data sets anyway, so it may not make a 
massive difference.

```R
s1k_tsne <- sneer(s1k, opt = "SPEC") 
```

The L-BFGS, CG and Spectral Directions methods are all gradient descent methods
that rely on a strong Wolfe line search to make progress. An alternative 
approach using an adaptive step size and a momentum scheme that emulates
the Nesterov Accelerated Gradient (NAG) is also available, which is closer to 
the original t-SNE optimization method in spirit, while being a bit more robust
when using other embedding methods.

A description of the connection between momentum and NAG is given in this 
[deep learning paper](www.jmlr.org/proceedings/papers/v28/sutskever13.pdf).
Additionally, it makes use of 
[adaptive restart](https://arxiv.org/abs/1204.3982). 

The step size is by the "bold driver" method, for which you can also use the 
`epsilon` parameter to set the initial learning rate.

```R
iris_tsne <- sneer(iris, epsilon = 10)
```

This method tends to be a little slower than the `"tsne"` optimization method, 
but it gives good results for most combinations of scaling, embedding method 
and data set.

But if in doubt, just use the default L-BFGS optimizer.

Previous: [Output Initialization](output-initialization.html). Next: [Embedding Methods](embedding-methods.html). Up: [Index](index.html).
