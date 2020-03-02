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
#include "Tests.h"

namespace Tests {

    vector<TestBundle> all_bundles() {
        vector<TestBundle> bundles;

        bundles.push_back(square());
        bundles.push_back(cos_1d_eq());
        bundles.push_back(cos_2d_eq());

        return bundles;
    }

    bool equality(VectorXd v1, VectorXd v2){
        if (v1.size() != v2.size()){
            return false;
        }
        for (int i = 0; i < v1.size(); ++i) {
            if (abs(v1(i)-v2(i)) > ATOL + RTOL * abs(v2(i))){
                return false;
            }
        }
        return true;
    }

    double func_rosenbrock(VectorXd v){
        // v dim is 3
        double x = v(0), y = v(1), z = v(2);
        // Newton cannot optimize this since it has no stationary point
        // but it's still nice to try out
        return 100*(y - x*x) + pow(1 - x, 2)
               + 100*(z - y*y) + pow(1 - y, 2);
    }

    static double func_cos_1d_eq(VectorXd v){
        // v dim is 1
        double x = v(0);
        // From https://www.wikiwand.com/en/Newton%27s_method#/Solution_of_cos(x)_=_x3
        // Models cos(x) - x^3 = 0
        // Integral of cos(x) - x^3 is sin(x) - 1/4 * x^4
        // When the derivative of sin(x) - 1/4 * x^4 = 0, we have
        // cos(x) - x^3 and hence the equation is solved
        return sin(x) - 0.25*x*x*x*x;
    }

    TestBundle cos_1d_eq(){
        VectorXd starting_x(1), target_x(1);
        starting_x << 3;
        target_x << 0.865474;
        return {"cos_1d_eq", func_cos_1d_eq, 20, starting_x, target_x};
    }

    static double func_cos_2d_eq(VectorXd v) {
        // v dim is 2
        double x = v(0), y = v(1);
        // Tried to make the problem a bit more interesting
        // and the solution is non trivial, nice !
        return y * (sin(x) - 0.25 * x * x * x * x);
    }

    TestBundle cos_2d_eq(){
        VectorXd starting_x(2), target_x(2);
        starting_x << 0.87, 10;
        target_x << 0, 0;
        return {"cos_2d_eq", func_cos_2d_eq, 50, starting_x, target_x};
    }

    static double func_square(VectorXd v) {
        // v dim is 2
        double x = v(0), y = v(1);
        // The humble 3D parabola
        return x*x + y*y;
    }

    TestBundle square(){
        VectorXd starting_x(2), target_x(2);
        starting_x << 1000, 1000;
        target_x << 0, 0;
        return {"square", func_square, 10, starting_x, target_x};
    }
}