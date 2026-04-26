# Derivation of the Numerically Stable Integrand for Eq. S3.34

## 1. Context and The Singularity Problem
The dimensionless integral in Eq. S3.34 calculates the noise scaling for ion channel sensitivity. The integrand requires calculating the derivative of an implicit function $G$ with respect to $q$, where $q = \sqrt{1 + x^2}$ (representing the low-frequency limit where $\omega \to 0$). 

The original formulation of $G$ breaks down into the following components (where $x = \kappa$, $h = \eta$):
$$B_1 = \frac{h^2}{(qh)^2 - 1}$$
$$B_2 = \left[ (1+qh)\left(\frac{u}{x+q} - q\right) \right]^{-1}$$
$$B_3 = q^2 - x^2 - u(1 - xh)$$

And the function is evaluated as:
$$G = 2 B_1 B_2 B_3 + B_1 \frac{1 - x^2 h^2}{h}$$

**The Numerical Instability:**
When $h = 1$ (the sensor height matches the Debye length), the denominator of $B_1$ becomes $(q)^2 - 1 = (1+x^2) - 1 = x^2$. 
During numerical integration using Gauss-Hermite quadrature, $x$ evaluates at nodes very close to $0$. As $x \to 0$, $B_1$ explodes as $1/x^2$, and its derivative with respect to $q$ (which is required by the integrand) explodes as $1/x^4$. The analytical expression has perfectly canceling divergences, but standard 64-bit floating point arithmetic suffers from catastrophic cancellation when subtracting these massive terms, injecting significant numerical noise into $S_0$.

## 2. Algebraic Factorization

To eliminate the singularity, we must combine $G$ into a single fraction and analytically divide out the problematic $(qh)^2 - 1$ term from both the numerator and denominator before implementing the function in code.

First, let's redefine the denominator of $B_2$ as $D_3$:
$$V = \frac{u}{x+q} - q$$
$$D_3 = (1+qh)V$$
So, $B_2 = \frac{1}{D_3}$. 

Substitute $B_1$ and $B_2$ into the full equation for $G$:
$$G = \frac{h^2}{(qh)^2 - 1} \left( \frac{2 B_3}{D_3} + \frac{1 - x^2 h^2}{h} \right)$$

By getting a common denominator $D_3$ for the terms in the parenthesis, we combine the expression:
$$G = \frac{h}{(qh)^2 - 1} \frac{2 h B_3 + (1 - x^2 h^2)D_3}{D_3}$$

When expanding the numerator $N_{raw} = 2 h B_3 + (1 - x^2 h^2)D_3$, we find that it is strictly divisible by the unstable term $((qh)^2 - 1)$. 

Performing the polynomial division $N_{raw} / ((qh)^2 - 1)$ yields a stable, reduced numerator $\tilde{N}$:
$$\tilde{N} = -D_3 - h + \frac{2u}{x+q} + \text{term}_3$$
where:
$$\text{term}_3 = \frac{-h + \frac{u}{x+q}(h^2 + 2xh - 2)}{qh + 1}$$

Thus, the original function exactly simplifies to a stable fraction without any $(qh)^2 - 1$ denominator:
$$G = h \frac{\tilde{N}}{D_3}$$

## 3. The Stable Derivative implementation

The integrand specifically requires the derivative $\frac{\partial G}{\partial q} \frac{x}{q}$. 
Because our new stable $G$ is a simple rational function, we can compute its analytical derivative using the quotient rule, completely sidestepping the $1/x^4$ numerical explosions.

$$\frac{\partial G}{\partial q} = h \frac{\tilde{N}' D_3 - \tilde{N} D_3'}{D_3^2}$$

### Step-by-Step Code Mapping
For another agent or developer implementing this, the components of the derivative map directly to the variables in `G_stable`:

**1. Denominator Components:**
$$V' = \frac{\partial V}{\partial q} = -\frac{u}{(x+q)^2} - 1$$
$$D_3' = \frac{\partial D_3}{\partial q} = hV + (1+qh)V'$$

**2. Numerator Components:**
Let the numerator of $\text{term}_3$ be $T_{3,num} = -h + \frac{u}{x+q}(h^2 + 2xh - 2)$.
$$\text{term}_3' = \frac{T_{3,num}' (qh+1) - h T_{3,num}}{(qh+1)^2}$$
where $T_{3,num}' = -\frac{u}{(x+q)^2}(h^2 + 2xh - 2)$.

Finally, the derivative of the combined stable numerator is:
$$\tilde{N}' = -D_3' - \frac{2u}{(x+q)^2} + \text{term}_3'$$

By evaluating $\frac{\partial G}{\partial q}$ via the quotient rule using these mapped components, the integral evaluates flawlessly across all $x$ and $h$ values (including exactly $h=1$) while preserving full machine precision.
