#include <Eigen/Core>
#include <Eigen/Geometry>

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/get_seconds.h>

#include "Skeleton.h"
#include "Bone.h"
#include "linear_blend_skinning.h"

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm> // for std::partial_sort

// Simple π constant
static const double PI = 3.14159265358979323846;

// 1. Fit a straight body chain (tail -> head) from sample verts
void build_body_joints_from_samples(
  const Eigen::MatrixXd & V,
  const std::vector<int> & sample_indices,
  int num_joints,
  std::vector<Eigen::Vector3d> & body_joints)
{
  const int S = (int)sample_indices.size();
  Eigen::MatrixXd P(S,3);
  for(int i=0;i<S;++i)
  {
    P.row(i) = V.row(sample_indices[i]);
  }

  // Center of the samples
  Eigen::Vector3d center = P.colwise().mean();

  // PCA: principal direction
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for(int i=0;i<S;++i)
  {
    Eigen::Vector3d d = P.row(i).transpose() - center;
    cov += d * d.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  Eigen::Vector3d axis = es.eigenvectors().col(2).normalized();

  // Project samples to axis to get min/max
  double t_min =  std::numeric_limits<double>::infinity();
  double t_max = -std::numeric_limits<double>::infinity();
  for(int i=0;i<S;++i)
  {
    double t = (P.row(i).transpose() - center).dot(axis);
    t_min = std::min(t_min,t);
    t_max = std::max(t_max,t);
  }

  // Evenly distribute joints along the axis
  body_joints.resize(num_joints);
  for(int i=0;i<num_joints;++i)
  {
    double alpha = (num_joints==1)?0.0:(double)i/(double)(num_joints-1);
    double t = t_min + alpha*(t_max - t_min);
    body_joints[i] = center + t*axis;
  }
}

// 2. Symmetric whiskers from head joint, perpendicular to body
void build_symmetric_whiskers(
  const Eigen::MatrixXd & V,
  const std::vector<int> & right_indices, // sample verts for right whisker
  const std::vector<int> & left_indices,  // sample verts for left whisker
  const Eigen::Vector3d & head_pos,
  const Eigen::Vector3d & body_axis,
  int num_joints_per_whisker,
  std::vector<Eigen::Vector3d> & right_joints,
  std::vector<Eigen::Vector3d> & left_joints)
{
  // Helper: average direction and length from samples
  auto compute_mean_dir_and_length = [&](const std::vector<int> & ids)
  {
    Eigen::Vector3d mean_dir = Eigen::Vector3d::Zero();
    double mean_len = 0.0;
    for(int idx : ids)
    {
      Eigen::Vector3d p = V.row(idx).transpose();
      Eigen::Vector3d d = p - head_pos;
      mean_dir += d;
      mean_len += d.norm();
    }
    if(!ids.empty())
    {
      mean_dir /= (double)ids.size();
      mean_len /= (double)ids.size();
    }

    // Remove body-axis component so whisker lies in perpendicular plane
    mean_dir -= mean_dir.dot(body_axis)*body_axis;
    if(mean_dir.norm() < 1e-8)
    {
      // Fallback perpendicular
      Eigen::Vector3d up(0,1,0);
      if(std::abs(up.dot(body_axis)) > 0.9) up = Eigen::Vector3d(0,0,1);
      mean_dir = up.cross(body_axis);
    }
    mean_dir.normalize();
    return std::make_pair(mean_dir,mean_len);
  };

  auto pr = compute_mean_dir_and_length(right_indices);
  auto pl = compute_mean_dir_and_length(left_indices);

  // Enforce symmetry: choose a direction and mirror it
  Eigen::Vector3d dir = pr.first - pl.first;
  if(dir.norm() < 1e-6) dir = pr.first;
  dir.normalize();

  Eigen::Vector3d right_dir = dir;
  Eigen::Vector3d left_dir  = -dir;

  double L = 0.5*(pr.second + pl.second);
  if(L <= 0.0) L = 0.2; // fallback length

  // Make whiskers slightly longer to exaggerate motion
  L *= 1.2;

  right_joints.resize(num_joints_per_whisker);
  left_joints.resize(num_joints_per_whisker);

  for(int i=0;i<num_joints_per_whisker;++i)
  {
    double s = (num_joints_per_whisker==1)?0.0:(double)i/(double)(num_joints_per_whisker-1);
    right_joints[i] = head_pos + s * L * right_dir;
    left_joints[i]  = head_pos + s * L * left_dir;
  }
}

// 3. Simple ragdoll (mass–spring system) on a generic joint set
struct Ragdoll
{
  // Node state
  Eigen::MatrixXd X;      // (#nodes,3) positions
  Eigen::MatrixXd V;      // (#nodes,3) velocities
  Eigen::VectorXd mass;   // (#nodes)

  // Springs
  Eigen::MatrixXi E;      // (#edges,2) indices
  Eigen::VectorXd rest_len; // (#edges)

  // Physical parameters
  double k_spring = 500.0;
  double damping  = 5.0;
  Eigen::Vector3d gravity = Eigen::Vector3d(0.0,-60.0,0.0); // stronger gravity
  double floor_y = -1.0;
  bool use_floor = true; // if false, ignore floor collision

  // Pinned nodes for interactive constraints
  std::vector<bool> pinned;
  Eigen::MatrixXd pinned_target; // same shape as X

  // Rest pose (for reset)
  Eigen::MatrixXd X_rest;
};

// Semi-implicit Euler mass-spring step
void step_ragdoll(Ragdoll & R, double dt, int substeps)
{
  if(substeps < 1) substeps = 1;
  if(dt <= 0.0) return;

  const int n = (int)R.X.rows();
  const int m = (int)R.E.rows();

  for(int s=0;s<substeps;++s)
  {
    // Accumulate forces
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n,3);

    // Gravity
    for(int i=0;i<n;++i)
    {
      F.row(i) += (R.mass(i) * R.gravity).transpose();
    }

    // Springs
    for(int e=0;e<m;++e)
    {
      int i = R.E(e,0);
      int j = R.E(e,1);
      Eigen::Vector3d xi = R.X.row(i).transpose();
      Eigen::Vector3d xj = R.X.row(j).transpose();
      Eigen::Vector3d d  = xj - xi;
      double len = d.norm();
      if(len < 1e-8) continue;
      Eigen::Vector3d dir = d/len;
      double stretch = len - R.rest_len(e);
      Eigen::Vector3d f = R.k_spring * stretch * dir;
      F.row(i) +=  f.transpose();
      F.row(j) += -f.transpose();
    }

    // Damping
    for(int i=0;i<n;++i)
    {
      F.row(i) += (-R.damping * R.V.row(i)).transpose();
    }

    // Semi-implicit Euler integration
    for(int i=0;i<n;++i)
    {
      if(R.pinned[i])
      {
        // Pinned node follows target exactly
        R.X.row(i) = R.pinned_target.row(i);
        R.V.row(i).setZero();
        continue;
      }

      Eigen::Vector3d a = F.row(i).transpose() / std::max(R.mass(i),1e-6);
      Eigen::Vector3d v = R.V.row(i).transpose();
      Eigen::Vector3d x = R.X.row(i).transpose();

      v += dt * a;
      x += dt * v;

      // Collision with invisible floor
      if(R.use_floor && x.y() < R.floor_y)
      {
        x.y() = R.floor_y;
        if(v.y() < 0.0) v.y() = -0.35 * v.y(); // small bounce
      }

      R.V.row(i) = v.transpose();
      R.X.row(i) = x.transpose();
    }
  }
}

// 4. Simple bone geometry helper (uses joint indices)
struct BoneGeom
{
  int tail; // joint index of bone tail
  int tip;  // joint index of bone tip
};

// 5. Build Skeleton + Bone list + skinning weights W
void build_skeleton_and_weights(
  const std::vector<Eigen::Vector3d> & joints_body,
  const std::vector<Eigen::Vector3d> & joints_left,
  const std::vector<Eigen::Vector3d> & joints_right,
  Skeleton & skeleton,
  std::vector<BoneGeom> & bones,
  Eigen::MatrixXd & W,
  const Eigen::MatrixXd & V)
{
  const int Nb = (int)joints_body.size();
  const int Nl = (int)joints_left.size();
  const int Nr = (int)joints_right.size();
  const int num_bones = (Nb-1) + (Nl-1) + (Nr-1);

  bones.clear();
  bones.reserve(num_bones);
  skeleton.clear();
  skeleton.reserve(num_bones);

  int bone_index = 0;

  auto add_bone = [&](const Eigen::Vector3d & tail,
                      const Eigen::Vector3d & tip,
                      int parent_id,
                      const std::string & name)
  {
    Eigen::Vector3d axis = tip - tail;
    double length = axis.norm();
    if(length < 1e-8) length = 1e-3;

    // Build rest_T: x-axis along bone, y/z orthogonal
    Eigen::Vector3d x = axis.normalized();
    Eigen::Vector3d up(0,1,0);
    if(std::abs(up.dot(x)) > 0.9) up = Eigen::Vector3d(0,0,1);
    Eigen::Vector3d y = up.cross(x);
    if(y.norm() < 1e-8)
    {
      up = Eigen::Vector3d(0,0,1);
      y = up.cross(x);
    }
    y.normalize();
    Eigen::Vector3d z = x.cross(y);

    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M.block<3,1>(0,0) = x;
    M.block<3,1>(0,1) = y;
    M.block<3,1>(0,2) = z;
    M.block<3,1>(0,3) = tail;
    Eigen::Affine3d rest_T(M);

    Bone b(parent_id,bone_index,rest_T,length);
    b.xzx.setZero();
    skeleton.push_back(b);

    BoneGeom g;
    g.tail = -1; // filled later
    g.tip  = -1;
    bones.push_back(g);

    ++bone_index;
  };

  // Body chain: bones [0 .. Nb-2]
  for(int i=0;i<Nb-1;++i)
  {
    const Eigen::Vector3d & tail = joints_body[i];
    const Eigen::Vector3d & tip  = joints_body[i+1];
    int parent_id = (i==0)?-1:(i-1);
    add_bone(tail,tip,parent_id,"body");
  }

  int head_bone_index = Nb-2; // last body bone

  // Left whisker bones
  const int left_start = bone_index;
  for(int i=0;i<Nl-1;++i)
  {
    const Eigen::Vector3d & tail = joints_left[i];
    const Eigen::Vector3d & tip  = joints_left[i+1];
    int parent_id = (i==0)?head_bone_index:(left_start + i - 1);
    add_bone(tail,tip,parent_id,"left_whisker");
  }

  // Right whisker bones
  const int right_start = bone_index;
  for(int i=0;i<Nr-1;++i)
  {
    const Eigen::Vector3d & tail = joints_right[i];
    const Eigen::Vector3d & tip  = joints_right[i+1];
    int parent_id = (i==0)?head_bone_index:(right_start + i - 1);
    add_bone(tail,tip,parent_id,"right_whisker");
  }

  // Assign BoneGeom joint indices: joint ordering is
  // [body joints][left joints][right joints]
  int node_offset = 0;
  auto assign_chain = [&](int num_joints, int bone_offset)
  {
    for(int i=0;i<num_joints-1;++i)
    {
      int b = bone_offset + i;
      bones[b].tail = node_offset + i;
      bones[b].tip  = node_offset + i + 1;
    }
    node_offset += num_joints;
  };
  assign_chain(Nb,0);
  assign_chain(Nl,left_start);
  assign_chain(Nr,right_start);

  // Skinning weights: inverse-distance-squared to bone centers
  const int nV = (int)V.rows();
  W.setZero(nV,num_bones);

  std::vector<Eigen::Vector3d> centers(num_bones);
  for(int b=0;b<num_bones;++b)
  {
    const BoneGeom & g = bones[b];
    Eigen::Vector3d tail, tip;
    int NbNl = Nb + Nl;
    if(g.tail < Nb)
      tail = joints_body[g.tail];
    else if(g.tail < NbNl)
      tail = joints_left[g.tail-Nb];
    else
      tail = joints_right[g.tail-NbNl];

    if(g.tip < Nb)
      tip = joints_body[g.tip];
    else if(g.tip < NbNl)
      tip = joints_left[g.tip-Nb];
    else
      tip = joints_right[g.tip-NbNl];

    centers[b] = 0.5*(tail+tip);
  }

  for(int vi=0;vi<nV;++vi)
  {
    Eigen::Vector3d p = V.row(vi).transpose();
    std::vector<double> invd(num_bones);
    for(int b=0;b<num_bones;++b)
    {
      double d = (p - centers[b]).norm();
      d = std::max(d,1e-5);
      invd[b] = 1.0/(d*d);
    }

    // Keep top-4 influences
    std::vector<int> idx(num_bones);
    for(int b=0;b<num_bones;++b) idx[b] = b;
    int K = std::min(4,num_bones);
    std::partial_sort(
      idx.begin(),idx.begin()+K,idx.end(),
      [&](int a,int c){return invd[a] > invd[c];});

    double sum = 0.0;
    for(int b=0;b<num_bones;++b) W(vi,b) = 0.0;
    for(int k=0;k<K;++k)
    {
      int b = idx[k];
      W(vi,b) = invd[b];
      sum += invd[b];
    }
    if(sum > 0.0)
    {
      for(int b=0;b<num_bones;++b) W(vi,b) /= sum;
    }
  }
}

// 6. Build FULL ragdoll for whole skeleton (body + whiskers)
void build_full_ragdoll(
  const std::vector<Eigen::Vector3d> & joints_body,
  const std::vector<Eigen::Vector3d> & joints_left,
  const std::vector<Eigen::Vector3d> & joints_right,
  const std::vector<BoneGeom> & bones,
  double floor_y,
  Ragdoll & R)
{
  const int Nb = (int)joints_body.size();
  const int Nl = (int)joints_left.size();
  const int Nr = (int)joints_right.size();
  const int N_nodes = Nb + Nl + Nr;

  R.X.resize(N_nodes,3);
  R.V.setZero(N_nodes,3);
  R.mass.resize(N_nodes);
  R.pinned.assign(N_nodes,false);
  R.pinned_target.resize(N_nodes,3);

  int k = 0;
  for(const auto & p : joints_body)
  {
    R.X.row(k) = p.transpose();
    R.mass(k) = 1.0; // heavier for body
    ++k;
  }
  for(const auto & p : joints_left)
  {
    R.X.row(k) = p.transpose();
    R.mass(k) = 0.4;
    ++k;
  }
  for(const auto & p : joints_right)
  {
    R.X.row(k) = p.transpose();
    R.mass(k) = 0.4;
    ++k;
  }

  R.X_rest = R.X;
  R.pinned_target = R.X;

  // Springs: one per bone + two extra to link whisker roots to head
  const int num_bones = (int)bones.size();
  const int extra_springs = 2; // head-leftRoot, head-rightRoot
  R.E.resize(num_bones + extra_springs,2);
  R.rest_len.resize(num_bones + extra_springs);

  for(int b=0;b<num_bones;++b)
  {
    R.E(b,0) = bones[b].tail;
    R.E(b,1) = bones[b].tip;
    Eigen::Vector3d xi = R.X.row(bones[b].tail).transpose();
    Eigen::Vector3d xj = R.X.row(bones[b].tip ).transpose();
    R.rest_len(b) = (xj - xi).norm();
  }

  int head_node   = Nb - 1;     // last body joint
  int left_root   = Nb;         // first left joint
  int right_root  = Nb + Nl;    // first right joint

  int e = num_bones;
  R.E(e,0) = head_node;
  R.E(e,1) = left_root;
  R.rest_len(e) = (R.X.row(head_node) - R.X.row(left_root)).norm();
  ++e;

  R.E(e,0) = head_node;
  R.E(e,1) = right_root;
  R.rest_len(e) = (R.X.row(head_node) - R.X.row(right_root)).norm();

  // Make rest lengths slightly shorter to create immediate tension
  R.rest_len *= 0.95;

  // Floor and physics parameters for fast falling ragdoll
  R.floor_y = floor_y;
  R.use_floor = true;
  R.gravity  = Eigen::Vector3d(0.0,-5000.0,0.0); // strong gravity
  R.k_spring = 500.0;                          // relatively stiff
  R.damping  = 5.0;                            // moderate damping

  // Give a small kick so it starts moving
  for(int i=0;i<N_nodes;++i)
  {
    R.V.row(i) = 0.08 * Eigen::RowVector3d::Random();
  }
}

// 7. Compute bone transforms from joints (generic for all frames)
void compute_bone_transforms_from_joints(
  const std::vector<Eigen::Vector3d> & joints_rest,
  const std::vector<Eigen::Vector3d> & joints_cur,
  const std::vector<BoneGeom> & bones,
  std::vector<Eigen::Affine3d,Eigen::aligned_allocator<Eigen::Affine3d>> & T)
{
  const int num_bones = (int)bones.size();
  T.resize(num_bones);

  for(int b=0;b<num_bones;++b)
  {
    const BoneGeom & g = bones[b];

    Eigen::Vector3d tail_rest = joints_rest[g.tail];
    Eigen::Vector3d tip_rest  = joints_rest[g.tip ];
    Eigen::Vector3d tail_cur  = joints_cur[g.tail];
    Eigen::Vector3d tip_cur   = joints_cur[g.tip ];

    Eigen::Vector3d v_rest = tip_rest - tail_rest;
    Eigen::Vector3d v_cur  = tip_cur  - tail_cur;
    if(v_rest.norm() < 1e-8 || v_cur.norm() < 1e-8)
    {
      // Degenerate case: just translate tail
      Eigen::Affine3d A = Eigen::Affine3d::Identity();
      A.translation() = tail_cur - tail_rest;
      T[b] = A;
      continue;
    }

    // Rotation: rotate v_rest onto v_cur
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(v_rest,v_cur);
    Eigen::Matrix3d Rmat = q.toRotationMatrix();

    Eigen::Affine3d A = Eigen::Affine3d::Identity();
    A.linear() = Rmat;
    A.translation() = tail_cur - Rmat * tail_rest;
    T[b] = A;
  }
}

// 8. Main: single-mode full ragdoll dragon
int main(int argc, char * argv[])
{
  igl::opengl::glfw::Viewer viewer;

  // Load dragon mesh
  Eigen::MatrixXd V,U;
  Eigen::MatrixXi F;
  std::string mesh_path = (argc>1)?argv[1]:"../data/dragon/dragon.obj";
  if(!igl::read_triangle_mesh(mesh_path,V,F))
  {
    std::cerr<<"Failed to read "<<mesh_path<<std::endl;
    return EXIT_FAILURE;
  }
  U = V;

  // Invisible floor slightly below dragon (only affects bones, not mesh)
  double min_y = V.col(1).minCoeff();
  double max_y = V.col(1).maxCoeff();
  double floor_y = min_y - 0.15*(max_y - min_y);

  // Sample indices from your Open3D picking logs
  // Body centerline (tail -> head)
  std::vector<int> body_ids = {
    16580,16670,16694,6000,5976,5940,5911,5866,6811,6776,6744,1851
  };
  // Right whisker 4 points
  std::vector<int> right_ids = {22076,21847,21875,21917};
  // Left whisker 4 points
  std::vector<int> left_ids  = {22549,22324,22352,22390};

  // Construct rest joints: body + symmetric whiskers
  std::vector<Eigen::Vector3d> joints_body;
  std::vector<Eigen::Vector3d> joints_left;
  std::vector<Eigen::Vector3d> joints_right;

  const int NUM_BODY_JOINTS    = 12; // -> 11 body bones
  const int NUM_WHISKER_JOINTS = 4;  // -> 3 bones per whisker

  // Fit straight body line
  build_body_joints_from_samples(
    V,body_ids,NUM_BODY_JOINTS,joints_body);

  // Body axis from tail to head
  Eigen::Vector3d body_axis =
    (joints_body.back() - joints_body.front()).normalized();

  // Build symmetric whiskers around head (head at last body joint)
  build_symmetric_whiskers(
    V,right_ids,left_ids,
    joints_body.back(),body_axis,
    NUM_WHISKER_JOINTS,
    joints_right,joints_left); // right first, left second

  // Global joint ordering: [body][left][right]
  const int Nb = (int)joints_body.size();
  const int Nl = (int)joints_left.size();
  const int Nr = (int)joints_right.size();
  const int N_nodes = Nb + Nl + Nr;

  std::vector<Eigen::Vector3d> joints_rest(N_nodes);
  {
    int k = 0;
    for(const auto & p : joints_body) joints_rest[k++] = p;
    for(const auto & p : joints_left) joints_rest[k++] = p;
    for(const auto & p : joints_right) joints_rest[k++] = p;
  }

  // Build Skeleton, bones and weights
  Skeleton skeleton;
  std::vector<BoneGeom> bones;
  Eigen::MatrixXd W;
  build_skeleton_and_weights(
    joints_body,joints_left,joints_right,
    skeleton,bones,W,V);

  const int num_bones = (int)bones.size();

  // Full ragdoll for entire dragon
  Ragdoll rag_full;
  build_full_ragdoll(
    joints_body,joints_left,joints_right,
    bones,floor_y,rag_full);

  // Current joints come entirely from rag_full.X
  std::vector<Eigen::Vector3d> joints_current = joints_rest;

  // Body joint indices are 0..Nb-1 (used for picking/dragging)
  std::vector<int> body_nodes_indices(Nb);
  for(int i=0;i<Nb;++i) body_nodes_indices[i] = i;

  // Viewer setup: mesh + skeleton
  const int model_id    = 0;
  const int skeleton_id = 1;
  viewer.append_mesh();
  viewer.selected_data_index = 0;

  viewer.data_list[model_id].set_mesh(U,F);
  viewer.data_list[model_id].set_face_based(true);
  viewer.data_list[model_id].show_faces = true;
  viewer.data_list[model_id].show_lines = false;

  viewer.data_list[skeleton_id].clear();
  viewer.data_list[skeleton_id].show_lines = true;
  viewer.data_list[skeleton_id].show_faces = false;
  viewer.data_list[skeleton_id].show_overlay_depth = false;

  viewer.core().animation_max_fps = 60.;
  viewer.core().is_animating = true;

  // Time tracking for stable dt
  double last_time = igl::get_seconds();

  // Helper: update viewer from joints_current
  auto update_view_from_joints = [&]()
  {
    // 1) Compute bone transforms from joints_current
    std::vector<Eigen::Affine3d,Eigen::aligned_allocator<Eigen::Affine3d>> T;
    compute_bone_transforms_from_joints(
      joints_rest,joints_current,bones,T);

    // 2) Skin the mesh
    linear_blend_skinning(V,skeleton,T,W,U);
    viewer.data_list[model_id].set_vertices(U);
    viewer.data_list[model_id].compute_normals();

    // 3) Skeleton edges (real bones)
    Eigen::MatrixXd P(N_nodes,3);
    for(int i=0;i<N_nodes;++i) P.row(i) = joints_current[i].transpose();

    Eigen::MatrixXi Evis(num_bones,2);
    for(int b=0;b<num_bones;++b)
      Evis.row(b) = Eigen::RowVector2i(bones[b].tail,bones[b].tip);

    Eigen::RowVector3d bone_color(0.1,0.9,0.2);
    viewer.data_list[skeleton_id].set_edges(P,Evis,bone_color);

    // 4) Body joint points (only body chain, no whiskers)
    Eigen::MatrixXd pts(body_nodes_indices.size(),3);
    Eigen::MatrixXd cols(body_nodes_indices.size(),3);
    Eigen::RowVector3d teal(0.56471,0.84706,0.76863);
    Eigen::RowVector3d pink(0.99608,0.76078,0.76078);
    for(int i=0;i<(int)body_nodes_indices.size();++i)
    {
      int idx = body_nodes_indices[i];
      pts.row(i) = joints_current[idx].transpose();

      bool pinned_flag = rag_full.pinned[idx];
      cols.row(i) = pinned_flag ? pink : teal;
    }
    viewer.data_list[skeleton_id].set_points(pts,cols);
  };

  // Mouse interaction: drag body nodes
  int selected_node = -1; // global joint index of selected body node
  int mouse_x = 0, mouse_y = 0;
  double mouse_z = 0.0;

  // Helper: pick body node in screen space
  auto pick_body_node = [&](igl::opengl::glfw::Viewer & v,
                            const Eigen::MatrixXd & body_positions)->int
  {
    Eigen::RowVector3f last_mouse(
      (float)mouse_x,
      (float)(v.core().viewport(3)-mouse_y),
      0.0f);

    // Project body joints to screen
    Eigen::MatrixXf P2d;
    igl::project(
      body_positions.cast<float>(),
      v.core().view,
      v.core().proj,
      v.core().viewport,
      P2d);

    // Find closest joint
    Eigen::VectorXf D = (P2d.rowwise()-last_mouse).rowwise().norm();
    int id;
    float minD = D.minCoeff(&id);
    if(minD < 30.0f)
    {
      mouse_z = P2d(id,2);
      return body_nodes_indices[id]; // global joint index
    }
    return -1;
  };

  viewer.callback_mouse_down =
    [&](igl::opengl::glfw::Viewer & v, int button, int)->bool
  {
    if(button != GLFW_MOUSE_BUTTON_LEFT) return false;

    // Use current ragdoll positions for picking
    Eigen::MatrixXd body_pos(Nb,3);
    for(int i=0;i<Nb;++i) body_pos.row(i) = rag_full.X.row(i);

    int picked = pick_body_node(v,body_pos);
    if(picked == -1) return false;

    selected_node = picked;

    // Pin in rag_full
    rag_full.pinned[selected_node] = true;
    rag_full.pinned_target.row(selected_node) = rag_full.X.row(selected_node);

    return true;
  };

  viewer.callback_mouse_up =
    [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    if(selected_node >= 0)
    {
      rag_full.pinned[selected_node] = false;
    }
    selected_node = -1;
    return false;
  };

  viewer.callback_mouse_move =
    [&](igl::opengl::glfw::Viewer & v, int x, int y)->bool
  {
    mouse_x = x; mouse_y = y;
    if(selected_node < 0) return false;

    Eigen::Vector3f drag_pos(
      (float)x,
      (float)(v.core().viewport(3)-y),
      (float)mouse_z);
    Eigen::Vector3f world;
    igl::unproject(
      drag_pos,
      v.core().view,
      v.core().proj,
      v.core().viewport,
      world);
    Eigen::Vector3d w = world.cast<double>();

    // Move pinned target for selected body joint
    rag_full.pinned_target.row(selected_node) = w.transpose();
    return true;
  };

  // Physics + rendering update per frame
  viewer.callback_pre_draw =
    [&](igl::opengl::glfw::Viewer &)->bool
  {
    double now = igl::get_seconds();
    double dt_real = now - last_time;
    if(dt_real < 0.0) dt_real = 0.0;
    if(dt_real > 0.1) dt_real = 0.1; // clamp
    last_time = now;

    // Use a small fixed dt for stability, scaled by animation flag
    const double dt_base = 1.0/120.0;
    const int    substeps = 5;
    double dt_phys = viewer.core().is_animating ? dt_base : 0.0;

    // Step full ragdoll (body + whiskers)
    step_ragdoll(rag_full,dt_phys,substeps);

    // Copy ragdoll joints into joints_current
    for(int i=0;i<N_nodes;++i)
    {
      joints_current[i] = rag_full.X.row(i).transpose();
    }

    // Update viewer from joints_current
    update_view_from_joints();
    return false;
  };

  // Keyboard: pause, reset, visibility toggles
  viewer.callback_key_pressed =
    [&](igl::opengl::glfw::Viewer & v,
        unsigned char key,
        int)->bool
  {
    switch(key)
    {
      // Toggle simulation play / pause (physics)
      case ' ':
        v.core().is_animating = !v.core().is_animating;
        break;

      // Reset ragdoll back to rest pose above floor
      case 'R':
      case 'r':
      {
        rag_full.X = rag_full.X_rest;
        rag_full.V.setZero();
        for(size_t i=0;i<rag_full.pinned.size();++i)
          rag_full.pinned[i] = false;
        selected_node = -1;

        // Also reset joints_current for immediate visual reset
        for(int i=0;i<N_nodes;++i)
          joints_current[i] = rag_full.X.row(i).transpose();
        break;
      }

      // Toggle mesh faces
      case 'M':
      case 'm':
        viewer.data_list[model_id].show_faces =
          !viewer.data_list[model_id].show_faces;
        break;

      // Toggle mesh wireframe
      case 'W':
      case 'w':
        viewer.data_list[model_id].show_lines =
          !viewer.data_list[model_id].show_lines;
        break;

      // Toggle skeleton visibility
      case 'S':
      case 's':
        viewer.data_list[skeleton_id].is_visible =
          !viewer.data_list[skeleton_id].is_visible;
        break;

      default:
        return false;
    }
    return true;
  };

  // Instructions
  std::cout << R"(
Dragon Ragdoll – Single Mode
----------------------------
This project shows a full ragdoll dragon:
  - Body and whiskers are simulated as a mass-spring system.
  - Only the bones (joints) collide with an invisible floor.
  - The mesh is never simulated directly; it is deformed by LBS from the bones.

Mouse:
  Left-click + drag   : grab a BODY joint (green point) and pull it around.
                        The grabbed joint turns pink (pinned to the mouse).
  Camera controls     : default libigl viewer (rotate, pan, zoom).

Keyboard:
  [Space]             : toggle simulation play / pause.
  R,r                 : reset ragdoll to the rest pose above the floor.
  M,m                 : toggle mesh faces.
  W,w                 : toggle mesh wireframe.
  S,s                 : toggle skeleton visibility.

Notes:
  - Only body joints (centerline of the dragon) are draggable.
  - Whisker bones participate in the same ragdoll system
    but cannot be directly dragged; they follow the springs.
  - The floor is invisible and only affects the bones, not the mesh.
)";

  viewer.launch();
  return EXIT_SUCCESS;
}
