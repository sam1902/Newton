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
#include "Newton.h"

namespace Newton {

    /* Computes only the ith component of the gradient */
    static double grad_apprx(const VectorXd& v, function<double(VectorXd)> func, int i){
        double eps = v(i)*sqrt(numeric_limits<double>::epsilon())*EPS_SCALING;
        if (abs(v(i)) <= numeric_limits<double>::epsilon()){
            eps = numeric_limits<double>::epsilon()*EPS_SCALING;
        }
        VectorXd x_more = v, x_less = v;
        // Add/Sub a diff to the component we want the grad of
        x_more(i) += eps;
        x_less(i) -= eps;
        // Central difference scheme to cancel one or more terms of the Taylor series
        return (func(x_more) - func(x_less))/(2*eps);
    }

    static VectorXd grad_apprx(const VectorXd& v, function<double(VectorXd)> func){
        int size = v.size();
        VectorXd g(size);
        for (int i = 0; i < size; ++i){
            g(i) = grad_apprx(v, func, i);
        }
        return g;
    }

    static MatrixXd hessian_apprx(const VectorXd& v, function<double(VectorXd)> func){
        MatrixXd hess = MatrixXd::Zero(v.size(), v.size());
        // Will compute df/didj for all pairs i j
        for (int i = 0; i < v.size(); ++i) {
            // [&] denote lexical closure hence the use of std::function instead
            // of function pointers (which can't store the extra context)

            // This lambda returns a scalar which is the eval of
            // the grad wrt to the i-th component of func at v2
            auto grad_xi = [&](const VectorXd& v2){
                return grad_apprx(v2, func, i);
            };
            // We then compute the gradient of the gradient of the i-th component along all other dims
            // hence having a vector of df/did1, df/did2, df/did3, df/did4, df/did5...
            // which we ofc put in the ith line of our hessian
            hess.row(i) = grad_apprx(v, grad_xi);
        }
        return hess;
    }

    function<VectorXd(VectorXd)> grad_apprx(function<double(VectorXd)> func){
        return [&](const VectorXd& v){
            return grad_apprx(v, func);
        };
    }

    function<MatrixXd(VectorXd)> hessian_apprx(function<double(VectorXd)> func){
        return [&](const VectorXd& v){
            return hessian_apprx(v, func);
        };
    }

    int newton(VectorXd& x, const function<double(VectorXd)>& func, int max_it){
        return newton(x, grad_apprx(func), hessian_apprx(func), max_it);
    }

    int newton(VectorXd& x, const function<VectorXd(VectorXd)>& grad, const function<MatrixXd(VectorXd)>& hess, int max_it){
        // x = current
        // A = hess
        // X = next
        // b = hess.x - grad(x)
        MatrixXd A;
        VectorXd b, gradVect = grad(x);
        int it = 0;
        if (max_it <= 0){
            max_it = INT_MAX;
        }
        while (gradVect.norm() > numeric_limits<double>::epsilon() && it < max_it){
            /* In theory, the following should work
             * x -= hess(x).inverse() * gradVect;
             * However, inversing the Hessian is often impossible because the determinant is almost zero
             * hence quite unstable (just print out hess(x).determinant()).
             * Using the .inverse() will work when optimising stuff like x^2+y^2, but not on
             * sin(x) - (x^4)/4 where it will just be zero.
             */
            A = hess(x);
            b = A*x - gradVect;
            x = A.colPivHouseholderQr().solve(b);
            gradVect = grad(x);
            it++;
        }
        return it;
    }
}