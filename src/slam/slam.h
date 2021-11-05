//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
/*!
\file    slam.h
\brief   SLAM Interface
\author  Joydeep Biswas, (C) 2018
*/
//========================================================================

#include <algorithm>
#include <vector>
#include <array>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

#ifndef SRC_SLAM_H_
#define SRC_SLAM_H_

using namespace std;
using std::round;
using Eigen::Vector2f;

namespace slam {


class SLAM {
 public:
  // Default Constructor.
  SLAM();
  ~SLAM();

  // Observe a new laser scan.
  void ObserveLaser(const std::vector<float>& ranges,
                    float range_min,
                    float range_max,
                    float angle_min,
                    float angle_max);

  // Observe new odometry-reported location.
  void ObserveOdometry(const Eigen::Vector2f& odom_loc,
                       const float odom_angle);

  // Get latest map.
  std::vector<Eigen::Vector2f> GetMap();

  // Get latest robot pose.
  void GetPose(Eigen::Vector2f* loc, float* angle) const;

 private:
  float GetMotionLikelihood(float x, float y, float a);
  void init();
  float GetObservationLikelihood(const vector<float>& ranges,
                                 float range_min,
                                 float range_max,
                                 float angle_min,
                                 float angle_max,
                                 float noise_x,
                                 float nosie_y,
                                 float noise_a);
  void UpdatePose(const vector<float>& ranges,
                  float range_min,
                  float range_max,
                  float angle_min,
                  float angle_max);
  void ReconstructMap(float lx, float ly);
  void UpdateLookupTable(const vector<float>& ranges,
                         float range_min,
                         float range_max,
                         float angle_min,
                         float angle_max);

  // Previous odometry-reported locations.
  bool odom_initialized_;
  Eigen::Vector2f tmp_pose_loc_;
  float tmp_pose_angle_;
  Eigen::Vector2f prev_pose_loc_;
  float prev_pose_angle_;
  Eigen::Vector2f cur_odom_loc_;
  float cur_odom_angle_;
  Eigen::Vector2f init_pose_loc_;
  float init_pose_angle_;

  // pose constraints
  const float POSE_MIN_DELTA_A = M_PI / 180.0 * 30.0;
  const float POSE_MIN_DELTA_D = 0.5;
  
  // table constraints
  constexpr static float NOISE_D_STEP = 0.05;
  constexpr static float NOISE_A_STEP = M_PI / 180.0 * 5.0;
  
  constexpr static float NOISE_X_BOUND = 0.5;
  constexpr static float NOISE_Y_BOUND = 0.5;
  constexpr static float NOISE_A_BOUND = M_PI / 180.0 * 30.0;

  constexpr static float HORIZON = 10.0;
  constexpr static float L_STEP = 0.05;
  constexpr static size_t L_WIDTH = (size_t) (2 * HORIZON / L_STEP) + 1;
  constexpr static size_t L_HEIGHT = (size_t) (2 * HORIZON / L_STEP) + 1;
  float** prev_prob_landmarks;
  bool prev_landmarks_initialized;

  constexpr static float S_RANGE = 5.0; // TODO: FIXME
  constexpr static size_t MASK_SIZE = (size_t) (2 * S_RANGE / L_STEP) + 1;
  float** prob_sensor;

  const float MIN_LOG_PROB = -10; // TODO: FIXME

  const float k_EPSILON = 1e-4;
  const int DOWNSAMPLE_RATE = 20;
  const Vector2f kLaserLoc = Vector2f(0.2, 0);
  vector<Vector2f> map_;

  size_t n_log;
  size_t n_lloc;
  size_t n_transformed_lloc;
};
}  // namespace slam

#endif   // SRC_SLAM_H_
