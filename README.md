# Netwon's method

<img src="https://upload.wikimedia.org/wikipedia/commons/e/e0/NewtonIteration_Ani.gif" alt="newton illustration" width="500">

This basic implementation of the Newton's method is designed to be reused transparently in future projects and hence be as robust as possible. 

## Basics
In a wide variety of cases, the goal is to optimise a continuously differentiable function of from R^n to R^m, if the function is convex, Newton's method is guarenteed to find the global optimum (and at least a local optimum otherwise) in O(n^2) time.

It works like this: Suppose you have a guess, `x_0` for the optimum, then you can update it like such:

```
    x_n+1 = x_n - f'(x_n)/f''(x_n)
```
It'll stabilize when it reaches a equilibrium where f'(x_n) is null, when the step difference is small enough you can stop.

The downside is that it's a second order method, hence require to compute the hessian which is costly. However, in many real life cases when you know the analytic expression of the function you're optimizing you can compute an analytic expression of the function's hessian making it super fast to compute at each step (sometimes it's even a constant !).

## Generality

This code is a bit more complicated because it handles the general case of a multivariable function (input is a vector) which also outputs a vector (possibly not the same size).


The update equation for the general case is a bit more complicated but analoguous to the unidimensional case (just replace the division by the second derivative by the inverse hessian, and the first derivative by the gradient at that point) \o/

