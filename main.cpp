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
#include <iostream>
#include <assert.h>
#include <Eigen/Dense>

#include "Newton.h"
#include "Tests.h"

// 0 : no verbose except what test is running
// 1 : a bit of verbose
// 2 : print grad and hessian at starting point
#define VERBOSE 1

using namespace std;
using namespace Newton;
using namespace Tests;

using Eigen::VectorXd;

int main() {

    auto bundles = all_bundles();
    for (TestBundle test : bundles){
        cout << "Running test \"" << test.name << "\"" << endl;
        function<double(VectorXd)> func = test.func;
        VectorXd x = test.starting_x;
        if (VERBOSE) {
            cout << "\tStarting x = [" << x.transpose() << "]" << endl;
            cout << "\tf(x) = " << func(x) << endl;
            if(VERBOSE > 1) {
                cout << "\tgrad(f(x)) = [" << grad_apprx(func)(x).transpose() << "]" << endl;
                cout << "\thess(f(x)) = " << endl << hessian_apprx(func)(x) << endl;
            }
        }

        int it = newton(x, func, max(test.max_it + 10, 1000));

        if (VERBOSE){
            cout << "\tTook " << it << " iterations" << endl;
            cout << "\tNewton result = [" << x.transpose() << "]" << endl;
            cout << "\tFunc = " << func(x) << endl;
            if(VERBOSE > 1) {
                cout << "\tGrad at result = [" << grad_apprx(func)(x).transpose() << "]" << endl;
                cout << "\tHess at result = " << endl << hessian_apprx(func)(x) << endl;
            }
        }

        // We shouldn't exceed the max it count !
        assert(it <= test.max_it);
        // X should be what we expect
        assert(equality(x, test.target_x));
        // Gradient should be null at stationnary point
        assert(equality(grad_apprx(func)(x), VectorXd::Zero(x.size())));
    }
    return 0;
}
