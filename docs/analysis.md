---
title: "Analysis"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Previous: [Exported Data](exported-data.html). Next: [Visualization](visualization.html). Up: [Index](index.html).

t-SNE is primarily a visualization technique, so the best way to analyze how
well the embedding has performed is to just look at the results. The 
[Visualization](visualization.html) section has a lot to say on that. But there 
have been some attempts to quantify the success of an embedding.

I will warn you ahead of time, these functions can be slow, because they 
compare every data point to every other data point multiple times. Try them
out on a small (100-1000) subset of your data first, and preferably on at 
least two sizes, so you can get a feel for both the absolute time it takes
and how they scale.

## `roc_auc_embed` and `pr_auc_embed`

When the data has obvious categories for each point (e.g. every observation in
the MNIST dataset is a digit from 0-9), you can treat every observation as if
it was a binary classifier. Once again using MNIST as an example, you would
hope that the close neighbors of an '8' digit are other '8' digits, both in
the input space and the output space.

Every point then brings with it its own classification result: rank all the 
other points by their distance, and treat every point in the same category as a 
positive classification, and every other point as a negative classification. 
You can now calculate the area under the ROC curve (ROC AUC) or the 
area under the Precision-Recall curve (PR AUC) for each point. The PR AUC has
become more popular recently because it is more sensitive to class imbalance,
which is likely to affect us if we use the embedding results in this way.

Two useful papers on this are 
[The relationship between Precision-Recall and ROC curves (PDF)](http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)
and
[Area under precision-recall curves for weighted and unweighted data](http://dx.doi.org/10.1371/journal.pone.0092209).

To calculate the ROC AUC and the PR AUC, you need to install and load the
[PRROC package](https://cran.r-project.org/package=PRROC):

```R
install.packages("PRROC")
library("PRROC")
```

Also, you must export the output distance matrix as part of the embedding
procedure:

```R
iris_tsne <- sneer(iris, ret = c("dy")) # "dy" means the output distance matrix
```

See the [Exported Data](exported-data.html) section for details on what extra
data can be exported and how.

Then, `sneer` package exports two functions to calculate the ROC AUC and PR AUC,
which need the distance matrix and a vector of factors that will identify each
point's category:

```R
pr_iris <- pr_auc_embed(iris_tsne$dy, iris$Species) # PR AUC data
roc_iris <- roc_auc_embed(iris_tsne$dy, iris$Species) # ROC AUC data
```

Both of these functions return a list with two items. `av_auc` contains a single
numeric value, which is the AUC averaged over every all the AUCs calculated
(one per observation). For more detail, there is the `label_av` item, which is
itself another list. This list provides the average AUC for every label. This
can help you detect if there is a particular class that isn't clustered very
well.

In terms of knowing how well you're doing: both the PR AUC and the ROC AUC
give a value of 1 for perfect recall. If the results are no better than random
the ROC AUC will be 0.5. For PR AUC, random results gives a number equal to
the proportion of those positive labels in the whole dataset, e.g. if you 
were using a category that is 10% of the data, a random PR AUC would be 0.1.
Therefore, if you have unequal class populations, the "random" `av_auc` value 
for PR AUC will be different for each data set.

You may also want to include the input distances `"dx"` in the `ret` argument
and run the same function on the input distance matrix. This should provide
an upper bound on how well the embedded results can do. But it is possible that
somehow the embedded results actually provide better clustering than the raw
data.

Using the embedding as a retrieval experiment and evaluating the PR AUC was
done in the 
[ws-SNE paper](http://jmlr.org/proceedings/papers/v32/yange14.html).

## `nbr_pres`

ROC AUC and PR AUC only works if you have some categories to classify each
point in the dataset into. An alternative approach would be to look at
the nearest neighbors of each point, in the input space and in the embedded
space. If they tend to have the same nearest neighbors, then the local
neighborhood around each point has been preserved, which is often all that you 
are interested in with probability-based embedding methods.

The neighbor preservation method quantifies this. You have to provide the
size of neighborhood, `k`, you're interested in (like the perplexity used in 
[Input Initialization](input-initialization.html)) 

To calculate the neighor preservation for a given `k`, first, make sure your
embedding result also returns both the input distance matrix (`dx`) and the
ouput distance matrix (`dy`). Then use the `nbr_pres` function:

```R
iris_tsne <- sneer(iris, ret = c("dx", "dy")) # dx is the input distance matrix
nbr25 <- nbr_pres(iris_tsne$dx, iris_tsne$dy, k = 25) # returns a vector
```

You will get back a vector containing the preservation for each point. 
It ranges from `0` (no neighbors in common) to `1` (all neighbors the same). 
With random performance, you'd expect a value around `k / (k - 1)`.

## `rnx_auc_embed`

Because `nbr_pres` only gives you the results for one neighborhood value, and 
it's hard to know what (if any) single value of `k` to use, an obvious 
extension is to use all possible values of `k` and then take a weighted average,
with higher weights for smaller neighborhoods. This is what the area under the
RNX curve (RNX AUC) does.

```R
iris_tsne <- sneer(iris, ret = c("dx", "dy"))
rnx_iris <- rnx_auc_embed(iris_tsne$dx, iris_tsne$dy)
```

For the RNX AUC, 1 is perfect neighborhood preservation, and 0 what you would
expect from random behavior. The `rnx_auc_embed` function returns the
average RNX AUC value over all observations.

The RNX AUC was used as an evaluation method in the 
[multiscale JSE](http://dx.doi.org/10.1016/j.neucom.2014.12.095) paper.

## `quality_measures`

If you know for sure you definitely want to calculate all these values, then
you can ask `sneer` to calculate them automatically after the embedding. Pass
a character vector to the `quality_measures` parameter containing one or more
of: `rocauc` (ROC AUC), `prauc` (PR AUC) or `rnxauc` (RNX AUC) (these can be
abbreviated):

```R
# ROC AUC, PR AUC and RNX AUC
s1k_tsne <- sneer(s1k, quality_measures = c("rocauc", "prauc", "rnxauc"))

# Just the PR AUC
s1k_tsne <- sneer(s1k, quality_measures = c("p"))
```

The average value will be logged to console and will be returned to as an item
on the return list as `av_roc_auc`, `av_pr_auc` and `av_rnx_auc`.

These are just average values across all the points, so if you want access
to more details, you should use the standalone functions. And bear in mind
that the bigger the dataset, the more time consuming it gets to make these
calculations (they are O(N^2) in complexity). This is probably best kept as a 
convenience for smaller data sets.

Previous: [Exported Data](exported-data.html). Next: [Visualization](visualization.html). Up: [Index](index.html).
