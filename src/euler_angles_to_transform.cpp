#include "euler_angles_to_transform.h"

Eigen::Affine3d euler_angles_to_transform(
  const Eigen::Vector3d & xzx)
{
  Eigen::Affine3d A = Eigen::Affine3d::Identity();
  Eigen::AngleAxisd R1(xzx(0), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd R2(xzx(1), Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd R3(xzx(2), Eigen::Vector3d::UnitX());
  A.linear() = (R3 * R2 * R1).toRotationMatrix();
  A.translation().setZero();
  return A;
}
