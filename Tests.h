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
#ifndef GENERALNEWTONMETHOD_TESTS_H
#define GENERALNEWTONMETHOD_TESTS_H
#include <vector>
#include <Eigen/Dense>
// Absolute and relative tolerance for equality check
#define ATOL 1e-8
#define RTOL 1e-5

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace Tests {

    struct TestBundle {
        string name;
        function<double(VectorXd)> func;
        int max_it;
        VectorXd starting_x;
        VectorXd target_x;
    };

    /**
     * Checks that |v1 - v2| <= atol + rtol * |v2| element-wise
     * @param v1 vector to compare
     * @param v2 vector to compare
     * @return true if formula is verified for each element, false otherwise.
     */
    bool equality(VectorXd v1, VectorXd v2);

    TestBundle cos_1d_eq();
    TestBundle cos_2d_eq();
    TestBundle square();

    vector<TestBundle> all_bundles();

}

#endif //GENERALNEWTONMETHOD_TESTS_H
