# Sampling based on Gaussian kernel density estimation

## requirements

pytorch

## Characteristic

Supports calculation of Gaussian kernel density estimation of any dimensional data, allowing samples to have gradients, so that the calculated entropy also has gradients.

Implemented entirely in torch, allowing for GPU acceleration, as kernel density estimation for high-dimensional data takes some time.

## Functional

`guassian_kde.py` is an implementation that supports high-dimensional Gaussian kernel density estimation.

`demo.ipynb` gives some demonstrations, including cases of one-dimensional and two-dimensional data, and visualization. The demo also provides a **sampling implementation**, which implements the following function: for some data samples, new sample points are sampled, and the newly added sample points maximize the entropy of the kernel density estimate of the data, that is, the kernel density function is close to a uniform distribution under extreme conditions. The calculation of entropy is based on the trapezoidal approximation method, which is also implemented in torch. Since the Gaussian kernel density estimation method can retain the gradient, gradient descent is used for optimization here.

## Visualization

For 1d data:

1 initd data and 3 sampling data. The addition of new sampling points results in a more uniform kernel density (maximum entropy as much as possible)

<img title="" src="image/kde_1d.png" alt="loading-ag-89" data-align="inline">

For 2d data:

![loading-ag-91](image/kde_2d.png)

## Improvement requirements

There may be some computational inefficiencies in the current implementation, and you can propose contributions.

## Principle

We have a current action set $A$. When we determine initial values of $a^c$, we merge it into the original action set to get an action set $A^c$. Let

$$
y_j = p(a_j)\log p(a_j), \quad j \in \{1, \ldots, |A^c|\}
$$

Approximate calculation of entropy

$$
\begin{aligned}
H(a) &=-\int p(a)\log p(a) \,\mathrm{d}a \\
&\approx-\sum_{j=1}^{m-1}\Big(\frac{a_j-a_{j-1}}{2}\Big)(y_j+y_{j-1})
\end{aligned}
$$

According to the Gaussian kernel density estimation method, for $p$, we have

$$
p(a_j) = \frac{1}{|A^c|h\sqrt{2\pi}}\sum^{|A^c|}_{i=1}\exp\Big\{-\frac{(a_j-a_i)^2}{2h^2}\Big\}
$$

$|A|$ represents the length of the set $A$. Finding the gradient is to find the gradient of $H$ with respect to $a^c$:

$$
\frac{\partial H}{\partial a^c}=-\sum_{j=1}^{m-1}\Big(\frac{a_j-a_{j-1}}{2}\Big)\Big(\frac{\partial y_j}{\partial a^c}+\frac{\partial y_{j-1}}{\partial a^c}\Big)
$$

Now consider $y_j$:

$$
\begin{aligned}
&y_j = p(a_j)\log p(a_j) \\
&\log p(a_j) = -\log(|A^c|h\sqrt{2\pi}) + \sum_{k=1}^{|A^c|}\Big[-\frac{(a_j-a_k)^2}{2h^2}\Big]
\end{aligned}
$$

Consider the differential form of $y_j$:

$$
\frac{\partial y_j}{\partial a^c} = \frac{\partial p(a_j)}{\partial a^c}\big[\log p(a_j)+1\big]
$$

where

$$
\begin{aligned}
\frac{\partial p(a_j)}{\partial a^c} 
&= -\frac{1}{|A^c|h\sqrt{2\pi}}\exp\Big\{-\frac{(a_j-a^c)^2}{2h^2}\Big\} \cdot \frac{a_j-a^c}{h^2} \\
&= -\frac{a_j-a^c}{|A^c|h^3\sqrt{2\pi}}\exp\Big\{-\frac{(a_j-a^c)^2}{2h^2}\Big\}
\end{aligned}
$$

Therefore, the differential of $y_j$ with respect to $a^c$ can be found:

$$
\begin{aligned}
\frac{\partial y_j}{\partial a^c} &= \frac{\partial p(a_j)}{\partial a^c}\big[\log p(a_j)+1\big] \\
&= -\frac{a_j-a^c}{|A^c|h^3\sqrt{2\pi}}\exp\Big\{-\frac{(a_j-a^c)^2}{2h^2}\Big\} \cdot \Big\{-\log(|A^c|h\sqrt{2\pi}) + \sum_{k=1}^{|A^c|}\Big[-\frac{(a_j-a_k)^2}{2h^2}\Big] + 1\Big\}
\end{aligned}
$$

The calculation of $\frac{\partial y_{j-1}}{\partial a^c}$ is similar. We can substitute them into $\frac{\partial H}{\partial a^c}$ to calculate the specific gradient. In summary, the gradient of the initial sampling action based on maximizing information entropy is calculable. Once the gradient value is calculated, it can be multiplied by the learning rate and added to the original $a^c$ for multiple iterations. Therefore, the sampling can be optimized.
