#include "fast_mass_springs_precomputation_dense.h"
#include "signed_incidence_matrix_dense.h"
#include <Eigen/Dense>

bool fast_mass_springs_precomputation_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::MatrixXd & M,
  Eigen::MatrixXd & A,
  Eigen::MatrixXd & C,
  Eigen::LLT<Eigen::MatrixXd> & prefactorization)
{
  const int n = static_cast<int>(V.rows());
  const int ne = static_cast<int>(E.rows());
  const int nb = static_cast<int>(b.size());
  const double h  = delta_t;
  const double h2 = h * h;
  const double inv_h2 = 1.0 / h2;
  const double w = 1e10; 
  // 1) rest lengths r
  r.resize(ne);
  for(int e = 0; e < ne; ++e)
  {
    const int i = E(e,0);
    const int j = E(e,1);
    r(e) = (V.row(i) - V.row(j)).norm();
  }

  // 2) mass matrix M (diagonal)
  M = Eigen::MatrixXd::Zero(n, n);
  for(int i = 0; i < n; ++i)
  {
    if(i < m.size())
    {
      M(i,i) = m(i);
    }
  }

  // 3) signed incidence matrix A
  signed_incidence_matrix_dense(n, E, A);

  // 4) selection matrix C for pinned vertices
  C = Eigen::MatrixXd::Zero(nb, n);
  for(int i = 0; i < nb; ++i)
  {
    const int vi = b(i);
    if(vi >= 0 && vi < n)
    {
      C(i, vi) = 1.0;
    }
  }

  // 5) build quadratic matrix Q = (1/h^2) M + k A^T A + w C^T C
  Eigen::MatrixXd AtA = A.transpose() * A;
  Eigen::MatrixXd CtC = C.transpose() * C;

  Eigen::MatrixXd Q = inv_h2 * M + k * AtA + w * CtC;

  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
