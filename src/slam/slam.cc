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
\file    slam.cc
\brief   SLAM Starter Code
\author  Joydeep Biswas, (C) 2019
*/
//========================================================================

#include <algorithm>
#include <cmath>
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "shared/math/geometry.h"
#include "shared/math/math_util.h"
#include "shared/util/timer.h"

#include "slam.h"

#include "vector_map/vector_map.h"

using namespace math_util;
using Eigen::Affine2f;
using Eigen::Rotation2Df;
using Eigen::Translation2f;
using Eigen::Vector2f;
using Eigen::Vector2i;
using std::cout;
using std::endl;
using std::string;
using std::swap;
using std::vector;
using vector_map::VectorMap;
using std::abs;

namespace slam {

SLAM::SLAM() :
    prev_odom_loc_(0, 0),
    prev_odom_angle_(0),
    odom_initialized_(false),
    has_new_pose(false) {}

void SLAM::GetPose(Eigen::Vector2f* loc, float* angle) const {
  // Return the latest pose estimate of the robot.
  *loc = prev_odom_loc_;
  *angle = prev_odom_angle_;
}

void SLAM::ObserveLaser(const vector<float>& ranges,
                        float range_min,
                        float range_max,
                        float angle_min,
                        float angle_max) {
  // A new laser scan has been observed. Decide whether to add it as a pose
  // for SLAM. If decided to add, align it to the scan from the last saved pose,
  // and save both the scan and the optimized pose.
  if (has_new_pose == true) {
    has_new_pose = false;
  } else {
    return;
  }
}

float getDist(const Vector2f& odom, const Vector2f& prev_odom) {
  return sqrt( pow(prev_odom.x() - odom.x(), 2) + pow(pre_odom.y() - odom.y()) );
}

void SLAM::ObserveOdometry(const Vector2f& odom_loc, const float odom_angle) {
  if (!odom_initialized_) {
    prev_odom_angle_ = odom_angle;
    prev_odom_loc_ = odom_loc;
    odom_initialized_ = true;
    has_new_pose = true;
    return;
  }
  // Keep track of odometry to estimate how far the robot has moved between 
  // poses.
  if ( abs(odom_angle - prev_odom_angle_) < MIN_DELTA_A|| getDist(odom_loc, prev_odom_loc_) < MIN_DELTA_D ) {
    prev_odom_angle_ = odom_angle;
    prev_odom_loc_ = odom_loc;
    has_new_pose = true;
  } 
}

vector<Vector2f> SLAM::GetMap() {
  vector<Vector2f> map;
  // Reconstruct the map as a single aligned point cloud from all saved poses
  // and their respective scans.
  return map;
}

}  // namespace slam

/**
 * How to construct the map from the lookup table?
 * Likelihood lookup table -- what should be the increment value? 
 * Do we need to calculate p(s_{i+1}|x_i, x_{i+1}, x_i) ?
 * What if we go to the same location again (moving in a circle? Do we want to update the lookup table 
 *    entry, or just add a new one for every new pose?
 * Do we discard old data? When, and by what criteria?
 * Are laser readings our landmarks in this context?
 *    If not, what are the labdmarks and how do we correlate the landmarks?
 * */
