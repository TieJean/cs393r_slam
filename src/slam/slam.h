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
  float calculateMotionLikelihood(float x, float y, float a);
  void init();

  // Previous odometry-reported locations.
  bool odom_initialized_;
  Eigen::Vector2f prev_pose_loc_;
  float prev_pose_angle_;
  Eigen::Vector2f cur_odom_loc_;
  float cur_odom_angle_;
  Eigen::Vector2f init_pose_loc_;
  float init_pose_angle_;
  
  // pose constraints
  constexpr static float MIN_DELTA_A = M_PI / 180.0 * 30.0; // 30 degrees translate to radians
  constexpr static float MIN_DELTA_D = 0.5;
  
  // table constraints
  constexpr static float DELTA_D_STEP = 0.05;
  constexpr static float DELTA_A_STEP = M_PI / 180.0 * 5.0;
  
  constexpr static float DELTA_A_BOUND = MIN_DELTA_A + 2 * DELTA_A_STEP;
  constexpr static float DELTA_X_BOUND = MIN_DELTA_D + 2 * DELTA_D_STEP;
  constexpr static float DELTA_Y_BOUND = MIN_DELTA_D + 2 * DELTA_D_STEP;

  constexpr static size_t SIZE_X = (size_t) ((DELTA_X_BOUND * 2.0) / DELTA_D_STEP) + 1;
  constexpr static size_t SIZE_Y = (size_t) ((DELTA_Y_BOUND * 2.0) / DELTA_D_STEP) + 1;
  constexpr static size_t SIZE_A = (size_t) ((DELTA_A_BOUND * 2.0) / DELTA_A_STEP) + 1;
  // float prob_motion[SIZE_X][SIZE_Y][SIZE_A];
  float*** prob_motion;

  constexpr static float HORIZON = 10.0;
  constexpr static float L_STEP = 0.05;
  constexpr static size_t L_WIDTH = (size_t) (2 * HORIZON / L_STEP) + 1;
  constexpr static size_t L_HEIGHT = (size_t) (2 * HORIZON / L_STEP) + 1;
  float** prev_prob_landmarks;
  // float prev_prob_landmarks[L_HEIGHT][L_WIDTH];
  bool prev_landmarks_initialized;

  constexpr static float S_RANGE = 1.0; 
  constexpr static size_t MASK_SIZE = (size_t) (2 * S_RANGE / L_STEP) + 1;
  // float prob_sensor[MASK_SIZE][MASK_SIZE] = {};
  float** prob_sensor;
  const float k_EPSILON = 1e-4;
  // const int DOWNSAMPLE_RATE = 10;
  const Vector2f kLaserLoc = Vector2f(0.2, 0);
  vector<Vector2f> map_;
};
}  // namespace slam

#endif   // SRC_SLAM_H_
