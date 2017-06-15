---
title: "Multi-scale Gradients"
date: "January 18, 2017"
output:
  html_document:
    theme: cosmo
---

Up: [Index](index.html)

In two papers (in 
[ESANN 2014 (PDF)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-64.pdf)
and [Neurocomputing](https://dx.doi.org/10.1016/j.neucom.2014.12.095)), 
Lee and co-workers introduced the concept of multi-scale embedding. This seeks
to answer the thorny question of "what perplexity value should I use?" by
combining the results from multiple perplexity values.

For the input space, choose a set of perplexity values (they suggest powers of 2
to cover local and global structure), calculate input probabilities for each
one, and then use an average of these probabilities as $P$.

For the output probabilities, you also construct an average $Q$ matrix. This 
should raise a question of how the different matrices that contribute to the
average $Q$ could differ: so know that this method isn't applicable to 
t-SNE, only to methods that use an exponential kernel in the output space
where the decay parameter, $\beta$, can be modified. Methods such as NeRV, JSE,
ASNE and SSNE fit the bill.

To allow multi-scaling to be applied to different methods, we need to work out
what the plug-in gradient would look like. The new addition is that we have
a new expression for the output probability, $\bar{Q}$, which is now the average
of $U$ other probability matrices, $Q_u$:

$$
\bar{q}_{ij} = \frac{1}{U}\sum_{u}^{U} q_{iju}
$$

The multi-scale papers use $L$ instead of $U$. In the gradient below, I'm 
already using $l$ in a summation index. $U$ was pretty much the next letter
spare in the alphabet.

Mercifully, the gradient is simply:

$$
\frac{\partial \bar{q}_{ij}}{\partial q_{kmu}} = \frac{1}{U}
$$
when $i = k$ and $j = m$ and zero for everything else.

There's a slight reworking of the normalization expression, which is now:

$$
q_{iju} = \frac{w_{iju}}{\sum_{km} w_{kmu}} = \frac{w_{iju}}{S_{u}}
$$

for pair-wise normalizations and:

$$
q_{iju} = \frac{w_{iju}}{\sum_{k} w_{iku}} = \frac{w_{iju}}{S_{iu}}
$$

for point-wise normalizations. The sum of weights now has an extra index to 
indicate which weight matrix the grand total (for pair-wise normalization)
or sum of row $i$ (for point-wise normalization) was used.

Let's revisit the original summation-crazy expression for the gradient that I
droned on at great length about [here](gradients.md), and add an appropriate
item into the chain to cover the creation of $\bar{Q}$ from $Q$:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial \bar{q}_{ij}}
  \sum_{stu}
  \frac{\partial \bar{q}_{ij}}{\partial q_{stu}}
  \sum_{klv}
  \frac{\partial q_{stu}}{\partial w_{klv}}
  \sum_{mn}
  \frac{\partial w_{klv}}{\partial f_{mn}}
  \sum_{pq}
  \frac{\partial f_{mn}}{\partial d_{pq}}
  \frac{\partial d_{pq}}{\partial \mathbf{y_h}}  
$$
For the summation involving $q_{stu}$, note that the summation index $u$ is 
from 1 to $U$. Similarly for $w_{klv}$, the summation index $v$ is also from 1 
to $U$. Every other index is still a sum over $N$, the number of points.

Focussing just on the new expression, we can see based on what we just
said about the gradient of the probability averaging expression, that
$\partial \bar{q}_{ij} / \partial q_{stu} = 0$ unless $s = k$ and $t = l$ and
additionally, from the discussion of the normalization expression,
$\partial q_{stu} / \partial w_{klv} = 0$ unless $u = v$.

At this point, the gradient now looks like:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial \bar{q}_{ij}}
  \sum_{klu}
  \frac{\partial \bar{q}_{ij}}{\partial q_{klu}}
  \frac{\partial q_{klu}}{\partial w_{klu}}
  \sum_{mn}
  \frac{\partial w_{klv}}{\partial f_{mn}}
  \sum_{pq}
  \frac{\partial f_{mn}}{\partial d_{pq}}
  \frac{\partial d_{pq}}{\partial \mathbf{y_h}}  
$$
We can insert the $1/U$ expression for the gradient of the average
probability:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial \bar{q}_{ij}}
  \sum_{klu}
  \frac{1}{U}
  \frac{\partial q_{klu}}{\partial w_{klu}}
  \sum_{mn}
  \frac{\partial w_{klu}}{\partial f_{mn}}
  \sum_{pq}
  \frac{\partial f_{mn}}{\partial d_{pq}}
  \frac{\partial d_{pq}}{\partial \mathbf{y_h}}  
$$
and moving the summation and constant about:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \frac{1}{U}
  \sum_{u}
  \sum_{ij} 
  \frac{\partial C}{\partial \bar{q}_{ij}}
  \sum_{kl}
  \frac{\partial q_{klu}}{\partial w_{klu}}
  \sum_{mn}
  \frac{\partial w_{klu}}{\partial f_{mn}}
  \sum_{pq}
  \frac{\partial f_{mn}}{\partial d_{pq}}
  \frac{\partial d_{pq}}{\partial \mathbf{y_h}}  
$$
Now everything after the sum over $u$ looks a lot like it did without 
multiscaling, give or take a $\bar{q}_{ij}$ here, and a $w_{klu}$ there. But we
can proceed with the gradient derivation with exactly the same steps, to get
to:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  \frac{2}{U} \sum_{ju} \left(
  k_{iju}
  +
  k_{jiu}
  \right)
\left(\mathbf{y_i} - \mathbf{y_j}\right)
$$
where for a point-wise normalization:
$$k_{iju} = 
\frac{1}{S_{iu}}
\left[
\frac{\partial C}{\partial \bar{q}_{ij}}
-
\sum_{k} \frac{\partial C}{\partial \bar{q}_{ik}} 
q_{iku}
\right]
\frac{\partial w_{iju}}{\partial f_{ij}}
$$

and for a pair-wise normalization:
$$k_{iju} = 
\frac{1}{S_{u}}
\left[
\frac{\partial C}{\partial \bar{q}_{ij}}
-
\sum_{kl} \frac{\partial C}{\partial \bar{q}_{kl}} 
q_{klu}
\right]
\frac{\partial w_{iju}}{\partial f_{ij}}
$$
Sadly, in the multi-scaling world, we can't carry out the extra cancellations 
we were able to do to get to the simpler t-SNE, SSNE and ASNE gradients that 
are seen in the non-multi-scaled literature. They rely on an equivalence between
$\bar{q}_{ij}$ and $q_{ijk}$ that no longer exists. 

But at least we have a recipe for adding multi-scaling to existing method 
gradients: create the stiffness matrix, $K_u$ for each weight matrix separately,
and then average them to get the multi-scaled stiffness matrix. Multiply by the 
displacement as usual and the multi-scaled gradient is yours.

Up: [Index](index.html)
