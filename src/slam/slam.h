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

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

#ifndef SRC_SLAM_H_
#define SRC_SLAM_H_

using std::round;

namespace slam {

class SLAM {
 public:
  // Default Constructor.
  SLAM();

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
  // Previous odometry-reported locations.
  bool odom_initialized_;
  Eigen::Vector2f prev_pose_loc_;
  float prev_pose_angle_;
  Eigen::Vector2f cur_pose_loc_;
  float cur_pose_angle_;
  
  // pose constraints
  const float MIN_DELTA_A = M_PI / 180.0 * 30.0; // 30 degrees translate to radians
  const float MIN_DELTA_D = 0.5;
  
  // table constraints
  const float DELTA_A_BOUND = MIN_DELTA_A + M_PI / 180.0 * 10.0;
  const float DELTA_X_BOUND = MIN_DELTA_D + 0.1;
  const float DELTA_Y_BOUND = MIN_DELTA_D + 0.1;
  
  const float DELTA_D_STEP = 0.05;
  const float DELTA_A_STEP = M_PI / 180.0 * 5.0;

  const size_t size_x = (size_t) round((DELTA_X_BOUND * 2.0) / DELTA_D_STEP) + 1;
  const size_t size_y = (size_t) round((DELTA_Y_BOUND * 2.0) / DELTA_D_STEP) + 1;
  const size_t size_a = (size_t) round((DELTA_A_BOUND * 2.0) / DELTA_A_STEP) + 1;
  
  float prob_motion[size_x][size_y][size_a];

  const float k_EPSILON = 1e-4;
};
}  // namespace slam

#endif   // SRC_SLAM_H_
