#include "end_effectors_objective_and_gradient.h"
#include "transformed_tips.h"
#include "kinematics_jacobian.h"
#include "copy_skeleton_at.h"
#include <iostream>
#include <cassert>

void end_effectors_objective_and_gradient(
  const Skeleton & skeleton,
  const Eigen::VectorXi & b,
  const Eigen::VectorXd & xb0,
  std::function<double(const Eigen::VectorXd &)> & f,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> & grad_f,
  std::function<void(Eigen::VectorXd &)> & proj_z)
{
  const int num_bones = static_cast<int>(skeleton.size());
  const int dim = num_bones * 3;
  assert(xb0.size() == b.size()*3);

  f = [=](const Eigen::VectorXd & A)->double
  {
    assert(A.size() == dim);
    Skeleton sk = copy_skeleton_at(skeleton, A);
    Eigen::VectorXd tips = transformed_tips(sk, b);
    Eigen::VectorXd diff = tips - xb0;
    return diff.squaredNorm();
  };

  grad_f = [=](const Eigen::VectorXd & A)->Eigen::VectorXd
  {
    assert(A.size() == dim);
    Skeleton sk = copy_skeleton_at(skeleton, A);
    Eigen::VectorXd tips = transformed_tips(sk, b);
    Eigen::VectorXd diff = tips - xb0;
    Eigen::MatrixXd J;
    kinematics_jacobian(sk, b, J);
    return 2.0 * J.transpose() * diff;
  };

  proj_z = [=](Eigen::VectorXd & A)
  {
    assert(static_cast<int>(skeleton.size())*3 == A.size());
    for(int i = 0; i < static_cast<int>(skeleton.size()); ++i)
    {
      for(int k = 0; k < 3; ++k)
      {
        int idx = 3*i + k;
        double v = A(idx);
        double vmin = skeleton[i].xzx_min(k);
        double vmax = skeleton[i].xzx_max(k);
        if(v < vmin) v = vmin;
        if(v > vmax) v = vmax;
        A(idx) = v;
      }
    }
  };
}
