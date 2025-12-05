#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  const int n  = static_cast<int>(V.rows());
  const int ne = static_cast<int>(E.rows());
  const int nb = static_cast<int>(b.size());
  const double h  = delta_t;
  const double h2 = h * h;
  const double inv_h2 = 1.0 / h2;
  const double w = 1e10; // pin penalty

  // 1) rest lengths
  r.resize(ne);
  for(int e = 0; e < ne; ++e)
  {
    const int i = E(e,0);
    const int j = E(e,1);
    r(e) = (V.row(i) - V.row(j)).norm();
  }

  // 2) mass matrix M (diagonal)
  {
    std::vector<Eigen::Triplet<double> > triplets;
    triplets.reserve(n);
    for(int i = 0; i < n; ++i)
    {
      if(i < m.size() && m(i) != 0.0)
      {
        triplets.emplace_back(i, i, m(i));
      }
    }
    M.resize(n, n);
    M.setFromTriplets(triplets.begin(), triplets.end());
  }

  // 3) incidence matrix A
  signed_incidence_matrix_sparse(n, E, A);

  // 4) selection matrix C
  {
    std::vector<Eigen::Triplet<double> > triplets;
    triplets.reserve(nb);
    for(int i = 0; i < nb; ++i)
    {
      const int vi = b(i);
      if(vi >= 0 && vi < n)
      {
        triplets.emplace_back(i, vi, 1.0);
      }
    }
    C.resize(nb, n);
    C.setFromTriplets(triplets.begin(), triplets.end());
  }

  // 5) Q = (1/h^2) M + k A^T A + w C^T C
  Eigen::SparseMatrix<double> AtA = A.transpose() * A;
  Eigen::SparseMatrix<double> CtC = C.transpose() * C;

  Eigen::SparseMatrix<double> Q(n, n);
  Q = inv_h2 * M + k * AtA + w * CtC;

  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
