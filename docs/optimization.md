---
title: "Optimization"
output: html_document
---

Previous: [Output Initialization](output-initialization.html). Next: [Embedding Methods](embedding-methods.html). Up: [Index](index.html).

I have exposed a lot of options to do with optimization, but I don't really
recommend using most of them, except out of mordant curiosity. Here are the 
ones that matter:

### `max_iter`

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
converge. But most of the time, if you're just trying to eyeball a data set,
don't be afraid to crank down `max_iter`.

### `tol`

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

### `opt`

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

The default optimizer uses the Nesterov Accelerated Gradient scheme. A good
description is given in this 
[deep learning paper](www.jmlr.org/proceedings/papers/v28/sutskever13.pdf).
Additionally, it makes use of 
[adaptive restart](https://arxiv.org/abs/1204.3982). Due to the rather 
aggressive momentum schedule, there's a "burn in" period of a couple of steps 
of plain steepest descent to establish a sensible step size and direction. 
Otherwise, a poor start can cause premature convergence.

The step size is by the "bold driver" method, for which you can also use the 
`epsilon` parameter to set the initial learning rate.

```R
iris_tsne <- sneer(iris, epsilon = 10)
```

This method tends to be a little slower than the `"tsne"` optimization method, 
because it calculates the cost at each step size but it gives good results for 
most combinations of scaling, embedding method and data set.

There are some other optimization methods available if you want to try them, but
they all have some slight drawbacks to them.

The L-BFGS method is used in the 
[JSE paper](http://dx.doi.org/10.1016/j.neucom.2012.12.036) and can be used
by setting `opt` to `"L-BFGS"`:

```R
s1k_tsne <- sneer(s1k, opt = "L-BFGS")
```

However, this relies on the `stats::optim` function in R, which disappears off
for several steps of iteration. During this time, code external to `optim` that
might depend on a particular iteration number won't trigger. So I advise against
using, for example, the `perp_scale` options described in the 
[Input Initialization](input-initialization.html) section with this optimizer.
Also, it just doesn't do that much better than simpler options.

Some further options are available to you if you install the 
[rconjgrad](https://github.com/jlmelville/rconjgrad) package:

```R
devtools::install_github("jlmelville/rconjgrad")
library("rconjgrad")
```

This provides access to a couple of line search routines, that allow for the
strong Wolfe conditions to be met (important for some optimization methods), 
one by More-Thuente and one by Rasmussen. If you want to try using the 
conjugate gradient optimization method (which is the method used in the 
[NeRV](http://www.jmlr.org/papers/v11/venna10a.html) paper, you can:

```R
# conjugate gradient with More-Thuente line search
s1k_tsne <- sneer(s1k, opt = "CG-MT") 
# conjugate gradient with Rasmussen line search
s1k_tsne <- sneer(s1k, opt = "CG-R") 
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
massive difference. If you want to try it, it also needs the `rconjgrad`
package to be loaded and is invoked similarly to the conjugate gradient 
optimizer:

```R
# spectral direction with More-Thuente line search
s1k_tsne <- sneer(s1k, opt = "SPEC-MT") 
# spectral direction with Rasmussen line search
s1k_tsne <- sneer(s1k, opt = "SPEC-R") 
```

But if in doubt, just use the default optimizer.

Previous: [Output Initialization](output-initialization.html). Next: [Embedding Methods](embedding-methods.html). Up: [Index](index.html).
