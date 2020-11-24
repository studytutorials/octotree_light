#include "se/octant_ops.hpp"
#include "se/octree.hpp"
#include "se/algorithms/balancing.hpp"
#include "se/functors/axis_aligned_functor.hpp"
#include "se/filter.hpp"
#include "se/io/octree_io.hpp"
#include "se/utils/math_utils.h"
#include "se/node.hpp"
#include "se/functors/data_handler.hpp"
#include <random>
#include <functional>
#include <gtest/gtest.h>
#include <vector>
#include <stdio.h>
#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"

template<typename T>
using VoxelBlockType = typename T::VoxelBlockType;

// Truncation distance and maximum weight
#define MAX_DIST 2.f
#define MAX_WEIGHT 5
#define MU 0.1f

// Fusion level,
// 0-3 are the according levels in the voxel_block
// 4 is multilevel fusion
#define SCALE 4

// Returned distance when ray doesn't intersect sphere
#define SENSOR_LIMIT 20

// Number of frames to move from start to end position
#define FRAMES 4

// Activate (1) and deactivate (0) depth dependent noise
#define NOISE 0

// Box intersection
#define RIGHT	0
#define LEFT	1
#define MIDDLE	2

struct camera_parameter {
public:
  camera_parameter() {};

  camera_parameter(Eigen::Vector4f K,
      const Eigen::Vector2i& image_res, const Eigen::Matrix4f& T_MC) :
    image_res_(image_res), T_MC_(T_MC) {
    K_ << K(0), 0   , K(2), 0,
          0   , K(1), K(3), 0,
          0   , 0   , 1   , 0,
          0   , 0   , 0   , 1;
  };

  float focal_length_pix() const {return K_(0,0);}
  void setPose(Eigen::Matrix4f T_MC) {T_MC_ = T_MC;}
  Eigen::Vector2i imageResolution() const {return image_res_;}
  Eigen::Matrix4f T_MC() const {return T_MC_;}
  Eigen::Vector3f t_MC() const {return T_MC_.topRightCorner(3,1);}
  Eigen::Matrix3f R_MC() const {return T_MC_.topLeftCorner(3,3);};
  Eigen::Matrix4f K() const {return K_;}

private:
  Eigen::Vector2i image_res_;
  Eigen::Matrix4f T_MC_;
  Eigen::Matrix4f K_;
};

struct ray {
public:
  ray(const camera_parameter& camera_parameter)
      : focal_length_pix_(camera_parameter.focal_length_pix()),
        offset_(camera_parameter.imageResolution() / 2),
        origin_(camera_parameter.t_MC()),
        R_MC_(camera_parameter.R_MC()),
        direction_(Eigen::Vector3f(-1.f,-1.f,-1.f)) {};

  void operator()(int pixel_x, int pixel_y) {
    direction_.x() = -offset_.x() + 0.5 + pixel_x;
    direction_.y() = -offset_.y() + 0.5 + pixel_y;
    direction_.z() = focal_length_pix_;
    direction_.normalize();
    direction_ = R_MC_ * direction_;
  };

  Eigen::Vector3f origin() const {return  origin_;}
  Eigen::Vector3f direction() const {return direction_;}

private:
  float focal_length_pix_;
  Eigen::Vector2i offset_;
  Eigen::Vector3f origin_;
  Eigen::Matrix3f R_MC_;
  Eigen::Vector3f direction_;
};

struct obstacle {
  virtual float intersect(const ray& ray) = 0;
  virtual ~obstacle() {};
};

struct sphere_obstacle : obstacle {
public:
  sphere_obstacle() {};
  sphere_obstacle(const Eigen::Vector3f& centre, const float radius)
  : centre_(centre), radius_(radius) {};

  sphere_obstacle(const camera_parameter& camera_parameter, const Eigen::Vector2f& centre_angle,
  float centre_distance, float radius)
  : radius_(radius) {
      Eigen::Matrix3f R_MC = camera_parameter.R_MC();
      Eigen::Vector3f t_MC = camera_parameter.t_MC();
      float dist_y = std::sin(centre_angle.x()) * centre_distance;
      float dist_x = std::cos(centre_angle.x()) * std::sin(centre_angle.y()) * centre_distance;
      float dist_z = std::cos(centre_angle.x()) * std::cos(centre_angle.y()) * centre_distance;
      Eigen::Vector3f dist(dist_x, dist_y, dist_z);
      centre_ = R_MC * dist + t_MC;
  };

  float intersect(const ray& ray) {
    float dist(SENSOR_LIMIT);
    Eigen::Vector3f oc = ray.origin() - centre_;
    float a = ray.direction().dot(ray.direction());
    float b = 2.0 * oc.dot(ray.direction());
    float c = oc.dot(oc) - radius_ * radius_;
    float discriminant = b * b - 4 * a * c;
    if (discriminant >= 0) {
      float dist_tmp = (-b - sqrt(discriminant)) / (2.0 * a);
      if (dist_tmp < dist)
        dist = dist_tmp;
    }
    return dist;
  };

  Eigen::Vector3f centre() {return centre_;}
  float radius() {return radius_;}

private:
  Eigen::Vector3f centre_;
  float radius_;
};

struct box_obstacle : obstacle {
public:
  box_obstacle() {};
  box_obstacle(const Eigen::Vector3f& centre, float depth_value, float width, float height)
  : centre_(centre), dim_(Eigen::Vector3f(depth_value, width, height)) {
    min_corner_ = centre - Eigen::Vector3f(depth_value, width, height);
    max_corner_ = centre + Eigen::Vector3f(depth_value, width, height);
  };

  box_obstacle(const Eigen::Vector3f& centre, const Eigen::Vector3f& dim)
  : centre_(centre), dim_(dim) {
    min_corner_ = centre - dim / 2;
    max_corner_ = centre + dim / 2;
  };

  float intersect(const ray& ray) {
    float dist(SENSOR_LIMIT);
    /*
    Fast Ray-Box Intersection
    by Andrew Woo
    from "Graphics Gems", Academic Press, 1990
    */
    int num_dim = 3;
    Eigen::Vector3f hit_point = -1 * Eigen::Vector3f::Ones();				/* hit point */
    {
      bool inside = true;
      Eigen::Vector3i quadrant;
      int which_plane;
      Eigen::Vector3f max_T;
      Eigen::Vector3f candidate_plane;

      /* Find candidate planes; this loop can be avoided if
         rays cast all from the eye(assume perpsective view) */
      for (int i = 0; i < num_dim; i++)
        if(ray.origin()[i] < min_corner_[i]) {
          quadrant[i] = LEFT;
          candidate_plane[i] = min_corner_[i];
          inside = false;
        }else if (ray.origin()[i] > max_corner_[i]) {
          quadrant[i] = RIGHT;
          candidate_plane[i] = max_corner_[i];
          inside = false;
        }else	{
          quadrant[i] = MIDDLE;
        }

      /* Ray origin inside bounding box */
      if(inside)	{
        return 0;
      }

      /* Calculate T distances to candidate planes */
      for (int i = 0; i < num_dim; i++)
        if (quadrant[i] != MIDDLE && ray.direction()[i] !=0.)
          max_T[i] = (candidate_plane[i]-ray.origin()[i]) / ray.direction()[i];
        else
          max_T[i] = -1.;

      /* Get largest of the max_T's for final choice of intersection */
      which_plane = 0;
      for (int i = 1; i < num_dim; i++)
        if (max_T[which_plane] < max_T[i])
          which_plane = i;

      /* Check final candidate actually inside box */
      if (max_T[which_plane] < 0.f) return dist;
      for (int i = 0; i < num_dim; i++)
        if (which_plane != i) {
          hit_point[i] = ray.origin()[i] + max_T[which_plane] * ray.direction()[i];
          if (hit_point[i] < min_corner_[i] || hit_point[i] > max_corner_[i])
            return dist;
        } else {
          hit_point[i] = candidate_plane[i];
        }

      dist = (hit_point - ray.origin()).norm();
      return dist;
    }
  };

  Eigen::Vector3f centre() {return centre_;}
  Eigen::Vector3f dim() {return dim_;}
  Eigen::Vector3f min_corner() {return min_corner_;}
  Eigen::Vector3f max_corner() {return max_corner_;}

private:
  Eigen::Vector3f centre_;
  Eigen::Vector3f dim_;
  Eigen::Vector3f min_corner_;
  Eigen::Vector3f max_corner_;
};

struct generate_depth_image {
public:
  generate_depth_image() {};
  generate_depth_image(float* depth_image, const std::vector<obstacle*>& obstacles)
      : depth_image_(depth_image), obstacles_(obstacles) {};

  void operator()(const camera_parameter& camera_parameter) {
    float focal_length_pix = camera_parameter.focal_length_pix();
    ray ray(camera_parameter);
    Eigen::Vector2i depth_image_res(camera_parameter.imageResolution());

    for (int x = 0; x < depth_image_res.x(); x++) {
      for (int y = 0; y < depth_image_res.y(); y++) {
        ray(x, y);
        float dist(SENSOR_LIMIT);
        for (std::vector<obstacle*>::iterator obstacle = obstacles_.begin(); obstacle != obstacles_.end(); ++obstacle) {
          float dist_tmp = (*obstacle)->intersect(ray);
          if (dist_tmp < dist)
            dist = dist_tmp;
        }

        float regularisation = std::sqrt(1 + se::math::sq(std::abs(x + 0.5 - depth_image_res.x() / 2) / focal_length_pix)
                                         + se::math::sq(std::abs(y + 0.5 - depth_image_res.x() / 2) / focal_length_pix));
        float depth_value = dist / regularisation;
        if(NOISE) {
          static std::mt19937 gen{1};
          std::normal_distribution<> noise(0, 0.004 * depth_value * depth_value);
          depth_image_[x + y * depth_image_res.x()] = depth_value + noise(gen);
        }
        else
          depth_image_[x + y * depth_image_res.x()] = depth_value;
      }
    }
  }

private:
  float* depth_image_;
  std::vector<obstacle*> obstacles_;

};

inline float compute_scale(const Eigen::Vector3f& point_C,
                           const Eigen::Vector3f& t_MC,
                           const float            scaled_pixel,
                           const float            voxel_dim) {
  const float block_dist = (voxel_dim * point_C - t_MC).norm();
  const float pixel_size = block_dist * scaled_pixel;
  int scale = std::min(std::max(0, int(log2(pixel_size / voxel_dim) + 1)), 3);
  return scale;
}

template <typename VoxelBlockT>
void propagateDown(VoxelBlockT* block, const int scale) {
  const Eigen::Vector3i block_coord = block->coordinates();
  const int block_size = VoxelBlockT::size_li;
  for(int voxel_scale = scale; voxel_scale > 0; --voxel_scale) {
    const int stride = 1 << voxel_scale;
    for (int z = 0; z < block_size; z += stride)
      for (int y = 0; y < block_size; y += stride)
        for (int x = 0; x < block_size; x += stride) {
          const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
          auto parent_data = block->data(parent_coord, voxel_scale);
          float delta_x = parent_data.x - parent_data.x_last;
          const int half_step = stride / 2;
          for(int k = 0; k < stride; k += half_step) {
            for(int j = 0; j < stride; j += half_step) {
              for(int i = 0; i < stride; i += half_step) {
                const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j , k);
                auto voxel_data = block->data(voxel_coord, voxel_scale - 1);
                if(voxel_data.y == 0) {
                  voxel_data.x  =  parent_data.x;
                  voxel_data.y  =  parent_data.y;
                  voxel_data.delta_y = parent_data.delta_y;
                } else {
                  voxel_data.x  =  std::max(voxel_data.x + delta_x, -1.f);
                  voxel_data.y  =  fminf(voxel_data.y + parent_data.delta_y, MAX_WEIGHT);
                  voxel_data.delta_y = parent_data.delta_y;
                }
                block->setData(voxel_coord, voxel_scale - 1, voxel_data);
              }
            }
          }
          parent_data.x_last = parent_data.x;
          parent_data.delta_y = 0;
          block->setData(parent_coord, voxel_scale, parent_data);
        }
  }
}

template <typename VoxelBlockT>
void propagateUp(VoxelBlockT* block, const int scale) {
  const Eigen::Vector3i block_coord = block->coordinates();
  const int block_size = VoxelBlockT::size_li;
  for(int voxel_scale = scale; voxel_scale < se::math::log2_const(block_size); ++voxel_scale) {
    const int stride = 1 << (voxel_scale + 1);
    for (int z = 0; z < block_size; z += stride)
      for (int y = 0; y < block_size; y += stride)
        for (int x = 0; x < block_size; x += stride) {
          const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);

          float mean = 0;
          int sample_count = 0;
          float weight = 0;
          for (int k = 0; k < stride; k += stride / 2)
            for (int j = 0; j < stride; j += stride / 2)
              for (int i = 0; i < stride; i += stride / 2) {
                auto child_data = block->data(voxel_coord + Eigen::Vector3i(i, j, k), voxel_scale);
                if (child_data.y != 0) {
                  mean += child_data.x;
                  weight += child_data.y;
                  sample_count++;
                }
              }

          auto parent_data = block->data(voxel_coord, voxel_scale + 1);

          if (sample_count != 0) {
            // Update TSDF value to mean of its children
            mean /= sample_count;
            parent_data.x = mean;
            parent_data.x_last = mean;
            // Update weight (round up if > 0.5, round down otherwise)
            weight /= sample_count;
            parent_data.y = ceil(weight);
          } else {
            parent_data = MultiresTSDF::VoxelType::initData();
          }
          parent_data.delta_y = 0;
          block->setData(voxel_coord, voxel_scale + 1, parent_data);
        }
  }
}

template <typename VoxelBlockT>
void foreach(float                                  voxel_dim,
             const std::vector<VoxelBlockT*>& active_list,
             const camera_parameter&                camera_parameter,
             float*                                 depth_image_data) {
  const int num_elem = active_list.size();
  for(int i = 0; i < num_elem; ++i) {
    VoxelBlockT* block = active_list[i];
    const Eigen::Vector3i block_coord = block->coordinates();
    const int block_size = VoxelBlockT::size_li;

    const Eigen::Matrix4f T_CM = (camera_parameter.T_MC()).inverse();
    const Eigen::Vector3f t_CM = se::math::to_translation(T_CM);
    const Eigen::Matrix4f K = camera_parameter.K();
    const Eigen::Vector2i depth_image_res = camera_parameter.imageResolution();
    const float scaled_pix = (camera_parameter.K().inverse() * (Eigen::Vector3f(1, 0 ,1) - Eigen::Vector3f(0, 0, 1)).homogeneous()).x();

    // Calculate the maximum uncertainty possible
    int scale = compute_scale((block_coord + Eigen::Vector3i::Constant(block_size / 2)).cast<float>(),
                               t_CM, scaled_pix, voxel_dim);
    if (SCALE != 4)
      scale = SCALE;
    float stride = std::max(int(pow(2,scale)),1);
    for(float z = stride / 2; z < block_size; z += stride) {
      for (float y = stride / 2; y < block_size; y += stride) {
        for (float x = stride / 2; x < block_size; x += stride) {
          const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
          const Eigen::Vector3f point_C = (T_CM * (voxel_dim * voxel_coord.cast<float>()).homogeneous()).head(3);
          auto parent_data = block->data(voxel_coord.cast<int>(), scale);
          if (point_C.z() < 0.0001f)
            continue;
          const Eigen::Vector3f pixel_homo = K.topLeftCorner<3, 3>() * point_C;
          const float inverse_depth_value = 1.f / pixel_homo.z();
          const Eigen::Vector2f pixel_f = Eigen::Vector2f(
              pixel_homo.x() * inverse_depth_value,
              pixel_homo.y() * inverse_depth_value);
          if (pixel_f.x() < 0.5f || pixel_f.x() > depth_image_res.x() - 1.5f ||
              pixel_f.y() < 0.5f || pixel_f.y() > depth_image_res.y() - 1.5f)
            continue;

          float depth_value = depth_image_data[int(pixel_f.x()) + depth_image_res.x() * int(pixel_f.y())];
          const float dist = (depth_value - point_C.z()) * std::sqrt( 1 + se::math::sq(point_C.x() / point_C.z()) + se::math::sq(point_C.y() / point_C.z()));
          if (dist > -MU) {
            const float sample = fminf(MAX_DIST, dist);

            // Make sure that the max weight isn't greater than MAX_WEIGHT (i.e. y + 1)
            parent_data.y = std::min(parent_data.y, MAX_WEIGHT - 1);

            // Update TSDF value
            parent_data.x = (parent_data.x * parent_data.y + sample) / (parent_data.y + 1);

            // Update weight
            parent_data.delta_y++;
            parent_data.y = parent_data.y + 1;

            block->setData(voxel_coord, scale, parent_data);
          }
        }
      }
    }

    propagateDown(block, scale);
    propagateUp(block, 0);
  }
}

template <typename T>
std::vector<VoxelBlockType<MultiresTSDF::VoxelType>*> buildActiveList(se::Octree<T>& map, const camera_parameter& camera_parameter, float voxel_dim, SensorImpl& sensor) {
  const se::PagedMemoryBuffer<VoxelBlockType<MultiresTSDF::VoxelType> >& block_buffer =
      map.pool().blockBuffer();
  for(unsigned int i = 0; i < block_buffer.size(); ++i) {
    block_buffer[i]->active(false);
  }

  const Eigen::Matrix4f T_CM = (camera_parameter.T_MC()).inverse();
  std::vector<VoxelBlockType<MultiresTSDF::VoxelType>*> active_list;
  auto in_frustum_predicate =
      std::bind(se::algorithms::in_frustum<VoxelBlockType<MultiresTSDF::VoxelType>>,
                std::placeholders::_1, voxel_dim, T_CM, sensor);
  se::algorithms::filter(active_list, block_buffer, in_frustum_predicate);
  return active_list;
}

class MultiscaleTSDFMovingCameraTest : public ::testing::Test {
protected:
  MultiscaleTSDFMovingCameraTest() :
    depth_image_res_(640, 480),
    focal_length_mm_(1.95),
    pixel_dim_(0.006),
    K_(525.f, 525.f, depth_image_res_.x() / 2, depth_image_res_.y() / 2),
    sensor_({depth_image_res_.x(), depth_image_res_.y(), false,
             0.f, 10.f,
             K_(0), K_(1), K_(2), K_(3),
             Eigen::VectorXf(0), Eigen::VectorXf(0)})

  {
    size_ = 512;                       // 512 x 512 x 512 voxel^3
    voxel_dim_ = 0.005;                // 5 mm/voxel
    dim_ = size_ * voxel_dim_;         // [m^3]
    octree_.init(size_, dim_);

    Eigen::Matrix4f T_MC = Eigen::Matrix4f::Identity();
    camera_parameter_ = camera_parameter(K_, depth_image_res_, T_MC);

    const int block_size = VoxelBlockType<MultiresTSDF::VoxelType>::size_li;
    for(int z = block_size / 2; z < size_; z += block_size) {
      for(int y = block_size / 2; y < size_; y += block_size) {
        for(int x = block_size / 2; x < size_; x += block_size) {
          const Eigen::Vector3i voxel_coord(x, y, z);
          allocation_list.push_back(octree_.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z()));
        }
      }
    }
    octree_.allocate(allocation_list.data(), allocation_list.size());

    // Generate depth image
    depth_image_data_ =
        (float*) malloc(sizeof(float) * depth_image_res_.x() * depth_image_res_.y());
  }

  float* depth_image_data_;
  Eigen::Vector2i depth_image_res_;
  const float focal_length_mm_;
  const float pixel_dim_;
  Eigen::Vector4f K_;
  camera_parameter camera_parameter_;

  typedef se::Octree<MultiresTSDF::VoxelType> OctreeT;
  OctreeT octree_;
  SensorImpl sensor_;
  int size_;
  float voxel_dim_;
  float dim_;
  std::vector<VoxelBlockType<MultiresTSDF::VoxelType>*> active_list_;
  generate_depth_image generate_depth_image_;

private:
  std::vector<se::key_t> allocation_list;
};

TEST_F(MultiscaleTSDFMovingCameraTest, SphereTranslation) {
  std::vector<obstacle*> spheres;

  // Allocate spheres in world frame
  spheres.push_back(new sphere_obstacle(voxel_dim_ * Eigen::Vector3f(size_ * 1 / 2, size_ * 1 / 2, size_ / 2), 0.5f));
  generate_depth_image_ = generate_depth_image(depth_image_data_, spheres);

  int frames = FRAMES;
  for (int frame = 0; frame < frames; frame++) {
    Eigen::Matrix4f T_MC = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R_BC;
    R_BC << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    Eigen::Matrix3f R_MB = Eigen::Matrix3f::Identity();

    T_MC.topLeftCorner<3,3>()  = R_MB * R_BC;

    T_MC.topRightCorner<3,1>() = (R_MB * Eigen::Vector3f(-(size_ / 2 + frame * size_ / 8), 0, size_ / 2)
        + Eigen::Vector3f(size_ / 2, size_ / 2, 0)) * voxel_dim_;

    camera_parameter_.setPose(T_MC);
    generate_depth_image_(camera_parameter_);
    active_list_ = buildActiveList(octree_, camera_parameter_, voxel_dim_, sensor_);
    foreach(voxel_dim_, active_list_, camera_parameter_, depth_image_data_);
    std::stringstream f;

    f << "./out/scale_"  + std::to_string(SCALE) + "-sphere-linear_back_move-" + std::to_string(frame) + ".vtk";

    save_3d_value_slice_vtk(octree_, f.str().c_str(),
                      Eigen::Vector3i(0, 0, octree_.size() / 2),
                      Eigen::Vector3i(octree_.size(), octree_.size(), octree_.size() / 2 + 1),
                      MultiresTSDF::VoxelType::selectNodeValue, MultiresTSDF::VoxelType::selectVoxelValue,
                      octree_.maxBlockScale());
  }

  for (std::vector<obstacle*>::iterator sphere = spheres.begin(); sphere != spheres.end(); ++sphere) {
    delete *sphere;
  }
  free(depth_image_data_);

}

TEST_F(MultiscaleTSDFMovingCameraTest, SphereRotation) {
  std::vector<obstacle*> spheres;

  // Allocate spheres in world frame
  sphere_obstacle* sphere_close = new sphere_obstacle(voxel_dim_
      * Eigen::Vector3f(size_ * 1 / 8, size_ * 2 / 3, size_ / 2), 0.3f);
  sphere_obstacle* sphere_far   = new sphere_obstacle(voxel_dim_
      * Eigen::Vector3f(size_ * 7 / 8, size_ * 1 / 3, size_ / 2), 0.3f);
  spheres.push_back(sphere_close);
  spheres.push_back(sphere_far);

  generate_depth_image_ = generate_depth_image(depth_image_data_, spheres);

  int frames = FRAMES;
  for (int frame = 0; frame < frames; frame++) {
    Eigen::Matrix4f T_MC = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f R_BC;
    R_BC << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    float angle = float(frame) / float(frames) * 2 * M_PI / 4 - 2 * M_PI / 8;
    Eigen::Matrix3f R_MB;
    R_MB <<  std::cos(angle), -std::sin(angle), 0,
        std::sin(angle),  std::cos(angle), 0,
        0,                0, 1;

    T_MC.topLeftCorner<3,3>()  = R_MB * R_BC;

    T_MC.topRightCorner<3,1>() = (R_MB * Eigen::Vector3f(-(size_ / 2 + 16 * size_ / 8), 0, size_ / 2) + Eigen::Vector3f(size_ / 2, size_ / 2, 0)) * voxel_dim_;

    camera_parameter_.setPose(T_MC);
    generate_depth_image_(camera_parameter_);
    active_list_ = buildActiveList(octree_, camera_parameter_, voxel_dim_, sensor_);
    foreach(voxel_dim_, active_list_, camera_parameter_, depth_image_data_);
    std::stringstream f;

    f << "./out/scale_"  + std::to_string(SCALE) + "-sphere-rotational_move-" + std::to_string(frame) + ".vtk";

    save_3d_value_slice_vtk(octree_, f.str().c_str(),
                      Eigen::Vector3i(0, 0, octree_.size() / 2),
                      Eigen::Vector3i(octree_.size(), octree_.size(), octree_.size() / 2 + 1),
                      MultiresTSDF::VoxelType::selectNodeValue, MultiresTSDF::VoxelType::selectVoxelValue,
                      octree_.maxBlockScale());
  }

  for (std::vector<obstacle*>::iterator sphere = spheres.begin(); sphere != spheres.end(); ++sphere) {
    delete *sphere;
  }
  free(depth_image_data_);

}

TEST_F(MultiscaleTSDFMovingCameraTest, BoxTranslation) {
  std::vector<obstacle*> boxes;

  // Allocate boxes in world frame
  boxes.push_back(new box_obstacle(voxel_dim_ * Eigen::Vector3f(size_ * 1 / 2, size_ * 1 / 4, size_ / 2), voxel_dim_
  * Eigen::Vector3f(size_ * 1 / 4, size_ * 1 / 4, size_ / 4)));
  boxes.push_back(new box_obstacle(voxel_dim_ * Eigen::Vector3f(size_ * 1 / 2, size_ * 3 / 4, size_ / 2), voxel_dim_
  * Eigen::Vector3f(size_ * 1 / 4, size_ * 1 / 4, size_ / 4)));
  generate_depth_image_ = generate_depth_image(depth_image_data_, boxes);

  int frames = FRAMES;
  for (int frame = 0; frame < frames; frame++) {
    Eigen::Matrix4f T_MC = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R_BC;
    R_BC << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    Eigen::Matrix3f R_MB = Eigen::Matrix3f::Identity();

    T_MC.topLeftCorner<3,3>()  = R_MB * R_BC;

    T_MC.topRightCorner<3,1>() = (R_MB * Eigen::Vector3f(-(size_ / 2 + frame * size_ / 8), 0, size_ / 2)
        + Eigen::Vector3f(size_ / 2, size_ / 2, 0)) * voxel_dim_;

    camera_parameter_.setPose(T_MC);
    generate_depth_image_(camera_parameter_);
    active_list_ = buildActiveList(octree_, camera_parameter_, voxel_dim_, sensor_);
    foreach(voxel_dim_, active_list_, camera_parameter_, depth_image_data_);
    std::stringstream f;

    f << "./out/scale_"  + std::to_string(SCALE) + "-box-linear_back_move-" + std::to_string(frame) + ".vtk";

    save_3d_value_slice_vtk(octree_, f.str().c_str(),
                      Eigen::Vector3i(0, 0, octree_.size() / 2),
                      Eigen::Vector3i(octree_.size(), octree_.size(), octree_.size() / 2 + 1),
                      MultiresTSDF::VoxelType::selectNodeValue, MultiresTSDF::VoxelType::selectVoxelValue,
                      octree_.maxBlockScale());
  }

  for (std::vector<obstacle*>::iterator box = boxes.begin(); box != boxes.end(); ++box) {
    delete *box;
  }
  free(depth_image_data_);

}

TEST_F(MultiscaleTSDFMovingCameraTest, SphereBoxTranslation) {
  std::vector<obstacle*> obstacles;

  // Allocate boxes in world frame
  obstacles.push_back(new box_obstacle(voxel_dim_ * Eigen::Vector3f(size_ * 1 / 2, size_ * 1 / 4, size_ / 2), voxel_dim_
  * Eigen::Vector3f(size_ * 1 / 4, size_ * 1 / 4, size_ / 4)));
  obstacles.push_back(new sphere_obstacle(voxel_dim_ * Eigen::Vector3f(size_ * 1 / 2, size_ * 1 / 2, size_ / 2), 0.5f));
  generate_depth_image_ = generate_depth_image(depth_image_data_, obstacles);

  int frames = FRAMES;
  for (int frame = 0; frame < frames; frame++) {
    Eigen::Matrix4f T_MC = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R_BC;
    R_BC << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    Eigen::Matrix3f R_MB = Eigen::Matrix3f::Identity();

    T_MC.topLeftCorner<3,3>()  = R_MB * R_BC;

    T_MC.topRightCorner<3,1>() = (R_MB * Eigen::Vector3f(-(size_ / 2 + frame * size_ / 8), 0, size_ / 2)
        + Eigen::Vector3f(size_ / 2, size_ / 2, 0)) * voxel_dim_;

    camera_parameter_.setPose(T_MC);
    generate_depth_image_(camera_parameter_);
    active_list_ = buildActiveList(octree_, camera_parameter_, voxel_dim_, sensor_);
    foreach(voxel_dim_, active_list_, camera_parameter_, depth_image_data_);
    std::stringstream f;

    f << "./out/scale_"  + std::to_string(SCALE) + "-sphere-and-box-linear_back_move-" + std::to_string(frame) + ".vtk";

    save_3d_value_slice_vtk(octree_, f.str().c_str(),
                      Eigen::Vector3i(0, 0, octree_.size() / 2),
                      Eigen::Vector3i(octree_.size(), octree_.size(), octree_.size() / 2 + 1),
                      MultiresTSDF::VoxelType::selectNodeValue, MultiresTSDF::VoxelType::selectVoxelValue,
                      octree_.maxBlockScale());
  }

  for (std::vector<obstacle*>::iterator obstacle = obstacles.begin(); obstacle != obstacles.end(); ++obstacle) {
    delete *obstacle;
  }
  free(depth_image_data_);
}
