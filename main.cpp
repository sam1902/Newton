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
#include <cmath>

#include <Eigen/Dense>

#define EPS 1e-4

using namespace std;
using Eigen::Matrix2d;
using Eigen::Vector2d;

using Eigen::MatrixXd;
using Eigen::VectorXd;

double f2(Vector2d v);
Vector2d f2_grad(Vector2d v);
Matrix2d f2_hess(Vector2d v);
VectorXd grad_apprx(VectorXd v, double (*func)(VectorXd));
VectorXd newton(VectorXd x, VectorXd (*grad)(VectorXd), MatrixXd (*hess)(VectorXd));
Vector2d newton2d(const Vector2d x, Vector2d (*grad)(Vector2d), Matrix2d (*hess)(Vector2d));
// x = current
// A = hess
// X = next
// b = hess.x - grad(x)

int main() {
    // x = A.colPivHouseholderQr().solve(b)
    Vector2d x;
    x << 10, -10;
    x = newton2d(x, &f2_grad, &f2_hess);
    cout << x << endl;
    cout << f2_grad(x) << endl;
    return 0;
}
Vector2d newton2d(const Vector2d x, Vector2d (*grad)(Vector2d), Matrix2d (*hess)(Vector2d)) {
    return (Vector2d)newton(x, (VectorXd(*)(VectorXd))grad, (MatrixXd(*)(VectorXd))hess);
}

VectorXd newton(VectorXd x, VectorXd (*grad)(VectorXd), MatrixXd (*hess)(VectorXd)){
    MatrixXd A;
    VectorXd b;
    int it = 0;
    while (f2_grad(x).norm() > EPS){
        A = f2_hess(x);
        b = A*x - f2_grad(x);
        x = A.colPivHouseholderQr().solve(b);
        it++;
    }
    cout << "Newton finished in it: " << it << endl;
    return x;
}

double f2(Vector2d v){
    return exp(v(0)+v(1)) + pow(v(0), 2) + 2*pow(v(1), 2);
}

VectorXd grad_apprx(VectorXd v, double (*func)(VectorXd)){
    int size = v.size();
    VectorXd g(size);
    VectorXd vdi_up, vdi_down;
    for (int i = 0; i < size; ++i) {
        vdi_up = v;
        vdi_down = v;
        vdi_up(i) += EPS;
        vdi_down(i) -= EPS;
        g(i) = (func(vdi_up) - func(vdi_down))/(2*EPS);
    }
    return g;
}

Vector2d f2_grad(Vector2d v){
    Vector2d g;
    g <<
        exp(v(0)+v(1)) + 2*v(0),
        exp(v(0)+v(1)) + 4*v(1);
    return g;
}

Matrix2d f2_hess(Vector2d v){
    Matrix2d hess;
/*
    hess <<
         exp(v(0)+v(1)) + 2,    // dxdx
         exp(v(0)+v(1)),        // dxdy
         exp(v(0)+v(1)),        // dydx
         exp(v(0)+v(1)) + 2;    // dydy
*/

     double dxdy = exp(v(0)+v(1));
     double dxdx = dxdy + 2;
     double dydx = dxdy, dydy = dxdx;
    hess << dxdx, dxdy, dydx, dydy;
    return hess;
}
/*

Matrix2d f2_hess_apprx(Vector2d v){
    grad()
}*/
