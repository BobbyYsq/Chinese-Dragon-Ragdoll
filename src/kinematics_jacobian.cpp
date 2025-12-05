#include "kinematics_jacobian.h"
#include "transformed_tips.h"
#include "copy_skeleton_at.h"
#include <iostream>
#include <cassert>

void kinematics_jacobian(
  const Skeleton & skeleton,
  const Eigen::VectorXi & b,
  Eigen::MatrixXd & J)
{
  const int num_bones = static_cast<int>(skeleton.size());
  const int num_angles = num_bones * 3;
  const int num_b = b.size();
  const int m = num_b * 3;

  J.resize(m, num_angles);

  if(num_bones == 0 || num_b == 0)
  {
    J.setZero();
    return;
  }

  Eigen::VectorXd A(num_angles);
  for(int i = 0; i < num_bones; ++i)
  {
    A.segment<3>(3*i) = skeleton[i].xzx;
  }

  Eigen::VectorXd base_tips = transformed_tips(skeleton, b);
  const double h = 1e-6;

  for(int j = 0; j < num_angles; ++j)
  {
    Eigen::VectorXd A_plus = A;
    A_plus(j) += h;
    Skeleton sk_plus = copy_skeleton_at(skeleton, A_plus);
    Eigen::VectorXd tips_plus = transformed_tips(sk_plus, b);
    J.col(j) = (tips_plus - base_tips) / h;
  }
}
