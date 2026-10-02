#ifndef EIGEN_HPP
#define EIGEN_HPP

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <Eigen/CholmodSupport>

#include "sparse.hpp"

typedef Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> EigenSolver;

class SymbolicFactorizer {
    EigenSolver cholesky;
    Eigen::SparseMatrix<double> previous;
    Eigen::VectorXd solve(Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b);
public:
    std::vector<Vec3> linear_solve_factor(const SpMat<Mat3x3>& A, const std::vector<Vec3>& b);
};

Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b);

std::vector<Vec3> linear_solve(const SpMat<Mat3x3>& A, const std::vector<Vec3>& b);

std::vector<double> linear_solve(const SpMat<double>& A, const std::vector<double>& b);

int eigenvalue_decomposition(int n, double* A, double* w, bool computeEigenvalues);

int singular_value_decomposition(int m, int n, double* A, double* s, double* U, double* Vt);

#endif