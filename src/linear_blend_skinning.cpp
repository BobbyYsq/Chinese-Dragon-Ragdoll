#include "linear_blend_skinning.h"

void linear_blend_skinning(
  const Eigen::MatrixXd & V,
  const Skeleton & skeleton,
  const std::vector<Eigen::Affine3d,Eigen::aligned_allocator<Eigen::Affine3d> > & T,
  const Eigen::MatrixXd & W,
  Eigen::MatrixXd & U)
{
  const int nV = static_cast<int>(V.rows());
  const int nBones = static_cast<int>(skeleton.size());
  U.resize(nV,3);

  for(int v = 0; v < nV; ++v)
  {
    Eigen::Vector3d pos = V.row(v).transpose();
    Eigen::Vector3d blended(0,0,0);
    double total_w = 0.0;

    for(int i = 0; i < nBones; ++i)
    {
      int wi = skeleton[i].weight_index;
      if(wi < 0 || wi >= W.cols()) continue;
      double w = W(v, wi);
      if(w == 0.0) continue;
      blended += w * (T[i] * pos);
      total_w += w;
    }

    if(total_w > 0.0)
    {
      U.row(v) = blended.transpose();
    }
    else
    {
      U.row(v) = V.row(v);
    }
  }
}
