/*
 * Copyright (c) 2020 Samuel Prevost.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GENERALNEWTONMETHOD_NEWTON_H
#define GENERALNEWTONMETHOD_NEWTON_H

// Scaling of numeric_limit::epsilon(), if below 12 the resulting epsilon
// tends to be too small for accurate gradient computation, but a scaling too big is
// risky too !
#define EPS_SCALING 18

#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace Newton {
    /**
     * Returns a function which approximates the gradient of func at the inputted point.
     *
     * @param func the function to approximate the gradient of
     * @return a function which approximate the gradient of the input
     */
    function<VectorXd(VectorXd)> grad_apprx(function<double(VectorXd)> func);

    /**
     * Returns a function which approximates the hessian of func at the inputted point.
     *
     * @param func the function to approximate the hessian of
     * @return a function which approximate the hessian of the input
     */
    function<MatrixXd(VectorXd)> hessian_apprx(function<double(VectorXd)> func);

    /**
     * Finds a stationary point of func that is closest to the inputted x.
     * x is modified in place at each iteration.
     *
     * @param x the stating point of the newton method which will update at each iteration.
     * @param func the function which to find a stationnary point of.
     * @param max_it the maximum number of iterations to perform, set to -1 to stop when the gradient is low enough.
     * @return the number of iterations that have been needed to find an acceptable stationary point.
     */
    int newton(VectorXd& x, const function<double(VectorXd)>& func, int max_it);

    /**
     * See int newton(VectorXd& x, function<double(VectorXd)> func) for complete doc
     * Instead of approximating the gradient and hessian functions, you can provide them directly if you
     * know their analytical expression, making the search way faster.
     *
     * @param x the stating point of the newton method which will update at each iteration.
     * @param grad the function used to compute the gradient at each step
     * @param hess the function used to compute the hessian matrix at each step
     * @param max_it the maximum number of iterations to perform, set to -1 to stop when the gradient is low enough.
     * @return the number of iterations that have been needed to find an acceptable stationary point.
     */
    int newton(VectorXd& x, const function<VectorXd(VectorXd)>& grad, const function<MatrixXd(VectorXd)>& hess, int max_it);
}

#endif //GENERALNEWTONMETHOD_NEWTON_H
