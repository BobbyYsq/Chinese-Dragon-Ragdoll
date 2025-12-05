#include "transformed_tips.h"
#include "forward_kinematics.h"

Eigen::VectorXd transformed_tips(
  const Skeleton & skeleton, 
  const Eigen::VectorXi & b)
{
  const int num_b = b.size();
  Eigen::VectorXd tips(3 * num_b);
  tips.setZero();

  std::vector<Eigen::Affine3d,Eigen::aligned_allocator<Eigen::Affine3d> > T;
  forward_kinematics(skeleton, T);

  for(int i = 0; i < num_b; ++i)
  {
    int bi = b(i);
    const Bone & bone = skeleton[bi];
    Eigen::Vector3d canonical_tip(bone.length, 0.0, 0.0);
    Eigen::Vector3d rest_tip = bone.rest_T * canonical_tip;
    Eigen::Vector3d posed_tip = T[bi] * rest_tip;
    tips.segment<3>(3*i) = posed_tip;
  }

  return tips;
}
