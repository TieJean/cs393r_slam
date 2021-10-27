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
using std::vector;

// CONFIG_FLOAT(GAMMA, "GAMMA");
CONFIG_FLOAT(SENSOR_STD_DEV, "SENSOR_STD_DEV");
CONFIG_FLOAT(D_SHORT, "D_SHORT");
CONFIG_FLOAT(D_LONG, "D_LONG");
CONFIG_FLOAT(P_OUTSIDE_RANGE, "P_OUTSIDE_RANGE");
// CONFIG_FLOAT(MOTION_X_STD_DEV, "MOTION_X_STD_DEV");
// CONFIG_FLOAT(MOTION_Y_STD_DEV, "MOTION_Y_STD_DEV");
// CONFIG_FLOAT(MOTION_A_STD_DEV, "MOTION_A_STD_DEV");
CONFIG_FLOAT(MOTION_DIST_K1, "MOTION_DIST_K1");
CONFIG_FLOAT(MOTION_DIST_K2, "MOTION_DIST_K2");
CONFIG_FLOAT(MOTION_A_K1, "MOTION_A_K1");
CONFIG_FLOAT(MOTION_A_K2, "MOTION_A_K2");
// CONFIG_FLOAT(MAX_D_DIST, "MAX_D_DIST");
// CONFIG_FLOAT(MAX_D_ANGLE, "MAX_D_ANGLE");

namespace slam {

// returns the motion model log likelihood in a Gaussian distribution
float calculateMotionLikelihood(float x, float y, float a) {
  // assume the dimensions are independent
  float d_d = sqrt(pow(x, 2) + pow(y, 2));
  float d_stddev = CONFIG_MOTION_DIST_K1 * d_d + CONFIG_MOTION_DIST_K2 * abs(a) ; // TODO: fix me
  float a_stddev = CONFIG_MOTION_A_K1 * d_d + CONFIG_MOTION_A_K2 * abs(a);
  a_stddev = a_stddev > DELTA_A_BOUND ? DELTA_A_BOUND : a_stddev;

  float log_x = - pow(x, 2) / pow(d_stddev, 2);
  float log_y = - pow(y, 2) / pow(d_stddev, 2);
  float log_a = - pow(a, 2) / pow(a_stddev, 2);
  return log_x + log_y + log_a;
}

SLAM::SLAM() :
    prev_pose_loc_(0, 0),
    prev_pose_angle_(0),
    cur_pose_loc_(0, 0),
    cur_pose_angle_(0),
    odom_initialized_(false),
    prev_landmarks_initialized(false) {
      
      // populates motion model table
      // prob_motion = (float*) malloc(sizeof(float) * SIZE_X * SIZE_Y * SIZE_A);
      // *(prob_motion + i * (SIZE_X*SIZE_Y) + j*SIZE_Y + k)

      // prob_motion = new float[SIZE_X][SIZE_Y][SIZE_A];
      for (size_t i = 0; i < SIZE_X; i++) {
        float d_x = -DELTA_X_BOUND + i * DELTA_D_STEP;
        for (size_t j = 0; i < SIZE_Y; j++) {
          float d_y = -DELTA_Y_BOUND + i * DELTA_D_STEP;
          for (size_t k = 0; k < SIZE_A; k++) {
            float d_a = -DELTA_A_BOUND + k * DELTA_A_STEP;
            // calculate motion model probability
            prob_motion[i][j][k] = calculateMotionLikelihood(d_x, d_y, d_a);
          }
        }
      }

      // populate the observation likelihood mask table
      size_t idx_mean_x = MASK_SIZE / 2;
      size_t idx_mean_y = MASK_SIZE / 2;
      for (size_t i = 0; i < MASK_SIZE; ++i) {
        for (size_t j = 0; j < MASK_SIZE; ++j) {
          float x = sqrt( pow((idx_mean_x - i) * L_STEP, 2) + pow((idx_mean_y - j) * L_STEP, 2) );
          if (x < -CONFIG_D_SHORT) {
            prob_sensor[i][j] = - pow(CONFIG_D_SHORT, 2) / pow(CONFIG_SENSOR_STD_DEV, 2);
          } else if (x > CONFIG_D_LONG) {
            prob_sensor[i][j] = - pow(CONFIG_D_LONG, 2) / pow(CONFIG_SENSOR_STD_DEV, 2);
          } else {
            prob_sensor[i][j] = - pow(x, 2) / pow(CONFIG_SENSOR_STD_DEV, 2);
          }
        }
      }
    }

SLAM::~SLAM() {
  // if (prob_motion != nullptr) {free(prob_motion);}
  // delete [] prob_motion;
  // if (prev_prob_landmarks != nullptr) {free(prev_prob_landmarks);}
  // if (prob_sensor != nullptr) {free(prob_sensor)}
}

void SLAM::GetPose(Eigen::Vector2f* loc, float* angle) const {
  // Return the latest pose estimate of the robot.
  *loc = prev_pose_loc;
  *angle = prev_odom_angle_;
}

float getDist(const Vector2f& odom, const Vector2f& prev_odom) {
  return sqrt( pow(prev_odom.x() - odom.x(), 2) + pow(pre_odom.y() - odom.y()) );
}

void SLAM::ObserveLaser(const vector<float>& ranges,
                        float range_min,
                        float range_max,
                        float angle_min,
                        float angle_max) {
  // A new laser scan has been observed. Decide whether to add it as a pose
  // for SLAM. If decided to add, align it to the scan from the last saved pose,
  // and save both the scan and the optimized pose.
  if ( abs(cur_pose_angle_ - prev_pose_angle_) < MIN_DELTA_A && getDist(cur_pose_loc_, prev_pose_loc_) < MIN_DELTA_D ) {return;}
  
  if (prev_landmarks_initialized) {
    
    // TODO: calculations with the prev lookup table
    
    // TODO: 
    // for each ray: (x', y') in the new laser frame
    // transform (x', y') to the prev laser frame

    // get p(s_{i+1}|x_i, x_{i+1}, s_i) from the lookup table
    // calculate p(x_{i+1}|x_i, u_i) from lookup table (delta_x, delta_y, delta_theta)
    // calculate && store current lookup tables
    // calculate max{ p(s_{i+1}|x_i, x_{i+1}, s_i)p(x_{i+1}|x_i, u_i) }
  }

  // reset the lookup table
  for (size_t i = 0; i < L_HEIGHT; ++i) {
    for (size_t j = 0; j < L_WIDTH; ++j) {
      prev_prob_landmarks[i][j] = k_EPSILON;
    }
  }

  // fill the lookup table
  float step_size = (angle_max - angle_min) / ranges.size();
  for (int i = 0; i < ranges.size(); i++) {
    float angle_i = angle_min + i * step_size;
    float range_i = ranges[i];
    if ( range_i > HORIZON - k_EPSILON  || range_i < range_min) {continue;}
    
    // get landmark position in laser frame
    float lx = range_i * cos(angle_i);
    float ly = range_i * sin(angle_i);
    
    size_t idx_x = (size_t) round((lx + HORIZON) / L_STEP);
    size_t idx_y = (size_t) round((ly + HORIZON) / L_STEP);

    size_t min_idx_x = idx_x - MASK_SIZE / 2;
    min_idx_x = min_idx_x < 0 ? 0 : min_idx_x;
    size_t max_idx_x = idx_x + MASK_SIZE / 2;
    max_idx_x = max_idx_x > L_HEIGHT - 1 ? L_HEIGHT - 1 : max_idx_x;
    size_t min_idx_y = idx_y - MASK_SIZE / 2;
    min_idx_y = min_idx_y < 0 ? 0 : min_idx_y;
    size_t max_idx_y = idx_y + MASK_SIZE / 2;
    max_idx_y = max_idx_y > L_WIDTH - 1 ? L_WIDTH - 1 : max_idx_y;
    
    // fill in range around this laser reading
    for (int x = min_idx_x; x <= max_idx_x; x+=L_STEP) {
      for (int y = min_idx_y; y <= max_idx_y; y+=L_STEP) {
        float p = prob_sensor[x - (idx_x - MASK_SIZE / 2)][y - (idx_y - MASK_SIZE / 2)];
        prev_prob_landmarks[x][y] = p > prev_prob_landmarks[x][y] ? p : prev_prob_landmarks[x][y];
      }
    }
  }
  
  prev_landmarks_initialized = true;
  
  prev_pose_angle_ = cur_pose_angle_;
  prev_pose_loc_ = cur_pose_loc_;
}

void SLAM::ObserveOdometry(const Vector2f& odom_loc, const float odom_angle) {
  if (!odom_initialized_) {
    prev_pose_angle = odom_angle;
    prev_pose_loc = odom_loc;
    odom_initialized_ = true;
    return;
  }
  // Keep track of odometry to estimate how far the robot has moved between 
  // poses.
  cur_pose_angle_ = odom_angle;
  cur_pose_loc_ = odom_loc;
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
