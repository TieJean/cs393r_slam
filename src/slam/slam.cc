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
#include "config_reader/config_reader.h"

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
using std::log;

CONFIG_FLOAT(SENSOR_STD_DEV, "SENSOR_STD_DEV");
CONFIG_FLOAT(D_SHORT, "D_SHORT");
CONFIG_FLOAT(D_LONG, "D_LONG");
CONFIG_FLOAT(P_OUTSIDE_RANGE, "P_OUTSIDE_RANGE");
CONFIG_FLOAT(MOTION_DIST_K1, "MOTION_DIST_K1");
CONFIG_FLOAT(MOTION_DIST_K2, "MOTION_DIST_K2");
CONFIG_FLOAT(MOTION_A_K1, "MOTION_A_K1");
CONFIG_FLOAT(MOTION_A_K2, "MOTION_A_K2");

namespace slam {

config_reader::ConfigReader config_reader_({"config/particle_filter.lua"});

SLAM::SLAM() :
    odom_initialized_(false), 
    prev_pose_loc_(0, 0),
    prev_pose_angle_(0),
    cur_odom_loc_(0, 0),
    cur_odom_angle_(0),
    prev_landmarks_initialized(false) {
      cout << "Start of constructor" << endl;

      // allocates space for all three lookup tables
      prev_prob_landmarks = new float*[L_HEIGHT];
      for (size_t i = 0; i < L_HEIGHT; ++i) {
        prev_prob_landmarks[i] = new float[L_WIDTH];
      }
      prob_sensor = new float*[MASK_SIZE];
      for (size_t i = 0; i < MASK_SIZE; ++i) {
        prob_sensor[i] = new float[MASK_SIZE];
      }
      prob_motion = new float**[SIZE_X];
      for (size_t i = 0; i < SIZE_X; ++i) {
        prob_motion[i] = new float*[SIZE_Y];
        for (size_t j = 0; j < SIZE_Y; ++j) {
          prob_motion[i][j] = new float[SIZE_A];
        }
      }
      // populates motion model table
      // TODO: optimize motion model: it's symmetric!
      // prob_motion = new float[SIZE_X][SIZE_Y][SIZE_A];
      for (size_t i = 0; i < SIZE_X; i++) {
        float d_x = -DELTA_X_BOUND + i * DELTA_D_STEP;
        for (size_t j = 0; j < SIZE_Y; j++) {
          float d_y = -DELTA_Y_BOUND + i * DELTA_D_STEP;
          for (size_t k = 0; k < SIZE_A; k++) {
            float d_a = -DELTA_A_BOUND + k * DELTA_A_STEP;
            // calculate motion model probability
            cout << i << j << k << endl;
            prob_motion[i][j][k] = calculateMotionLikelihood(d_x, d_y, d_a);
          }
        }
      }

      cout << "adding observation likeilhood table" << endl;
      // prob_sensor = new float[MASK_SIZE][MASK_SIZE];
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
    cout << "end of constructor" << endl;
    }

SLAM::~SLAM() {
  for (size_t i = 0; i < L_HEIGHT; ++i) {
    delete[] prev_prob_landmarks[i];
  }
  delete[] prev_prob_landmarks;

  for (size_t i = 0; i < SIZE_X; ++i) {
    for (size_t j = 0; j < SIZE_Y; ++j) {
      delete[] prob_motion[i][j];
    }
    delete[] prob_motion[i];
  }
  delete[] prob_motion;

  for (size_t i = 0; i < MASK_SIZE; ++i) {
    delete[] prob_sensor[i];
  }
  delete[] prob_sensor;

}

// Returns the motion model log likelihood in a Gaussian distribution
float SLAM::calculateMotionLikelihood(float x, float y, float a) {
  // assume the dimensions are independent
  float d_d = sqrt(pow(x, 2) + pow(y, 2));
  float d_stddev = CONFIG_MOTION_DIST_K1 * d_d + CONFIG_MOTION_DIST_K2 * abs(a) ; // TODO: fix me
  float a_stddev = CONFIG_MOTION_A_K1 * d_d + CONFIG_MOTION_A_K2 * abs(a);
  a_stddev = a_stddev > DELTA_A_BOUND ? DELTA_A_BOUND : a_stddev;

  float log_x = - pow(x, 2) / pow(d_stddev, 2);
  float log_y = - pow(y, 2) / pow(d_stddev, 2);
  float log_a = - pow(a, 2) / pow(a_stddev, 2);
  cout << "End of motion likehood calc" << endl;
  return log_x + log_y + log_a;
}

void SLAM::GetPose(Eigen::Vector2f* loc, float* angle) const {
  // Return the latest pose estimate of the robot.
  *loc = prev_pose_loc_;
  *angle = prev_pose_angle_;
}

float getDist(const Vector2f& odom, const Vector2f& prev_odom) {
  return sqrt( pow(prev_odom.x() - odom.x(), 2) + pow(prev_odom.y() - odom.y(), 2) );
}

/*
 * loc_ab and angle_ab is the position of frame A in frame B.
 * loc_pa and angle_pa is the position of the object p in frame A.
 * Calculates the position of p in frame B.
 */
void TransformAToB(const Vector2f& loc_ab, float angle_ab,
                   const Vector2f& loc_pa, float angle_pa,
                   Vector2f* loc_ptr, float* angle_ptr) {
  float& new_angle = *angle_ptr;
  Vector2f& new_loc = *loc_ptr;
  
  Rotation2Df r(angle_ab);
  new_loc = loc_ab + r * loc_pa;
  new_angle = angle_ab + angle_pa;
}

void SLAM::ObserveLaser(const vector<float>& ranges,
                        float range_min,
                        float range_max,
                        float angle_min,
                        float angle_max) {
  // A new laser scan has been observed. Decide whether to add it as a pose
  // for SLAM. If decided to add, align it to the scan from the last saved pose,
  // and save both the scan and the optimized pose.
  if ( abs(cur_odom_angle_ - prev_pose_angle_) < MIN_DELTA_A && getDist(cur_odom_loc_, prev_pose_loc_) < MIN_DELTA_D ) {return;}
  // TODO: check if we need to construct map on the initial pose
  
  if (prev_landmarks_initialized) {
    float max_p = log(0.1*k_EPSILON);
    float max_p_dx = 0.0;
    float max_p_dy = 0.0;
    float max_p_da = 0.0;
  
    // check all possible poses of the car
    for (float delta_x = -DELTA_X_BOUND; delta_x < DELTA_X_BOUND; delta_x += DELTA_D_STEP) {
      for (float delta_y = -DELTA_Y_BOUND; delta_y < DELTA_Y_BOUND; delta_y += DELTA_D_STEP) {
        for (float delta_a = -DELTA_A_BOUND; delta_a < DELTA_A_BOUND; delta_a += DELTA_A_STEP) {
          Vector2f loc_ab(delta_x, delta_y);
          float angle_ab = delta_a;

          // calculate p(x_{i+1}|x_i, u_i) from motion lookup table
          size_t motion_idx_x = (size_t) round((delta_x + DELTA_X_BOUND) / DELTA_D_STEP);
          size_t motion_idx_y = (size_t) round((delta_y + DELTA_Y_BOUND) / DELTA_D_STEP);
          size_t motion_idx_a = (size_t) round((delta_a + DELTA_A_BOUND) / DELTA_D_STEP);
          float p_motion = prob_motion[motion_idx_x][motion_idx_y][motion_idx_a];

          // check all observations by using the lookup table from prev pose
          // TODO: Try to reduce duplicate calculations here if slow
          float step_size = (angle_max - angle_min) / ranges.size();
          for (size_t i = 0; i < ranges.size(); i++) {
            float angle_i = angle_min + i * step_size;
            float range_i = ranges[i];
            if ( range_i > HORIZON - k_EPSILON  || range_i < range_min) {continue;}

            // get landmark position in the new laser frame
            Vector2f loc_pa(range_i * cos(angle_i), range_i * sin(angle_i));
            float angle_pa = angle_i;
            
            // transform each ray in the new laser frame to the prev laser frame
            Vector2f transformed_lloc(0,0);
            float transformed_langle = 0.0;
            TransformAToB(loc_ab, angle_ab, loc_pa, angle_pa, &transformed_lloc, &transformed_langle);
            
            // get p(s_{i+1}|x_i, x_{i+1}, s_i) from the lookup table
            size_t landmark_idx_x = (size_t) round((transformed_lloc.x() + HORIZON) / L_STEP);
            size_t landmark_idx_y = (size_t) round((transformed_lloc.y() + HORIZON) / L_STEP);
            float p_landmark = prev_prob_landmarks[landmark_idx_x][landmark_idx_y];
            
            // calculate p(s_{i+1}|x_i, x_{i+1}, s_i)p(x_{i+1}|x_i, u_i)
            float prob = p_motion + p_landmark;
            if (prob > max_p) {
              max_p = prob;
              max_p_dx = delta_x;
              max_p_dy = delta_y;
              max_p_da = delta_a;
            }
          }
        }
      }
    }
    prev_pose_angle_ += max_p_da;
    prev_pose_loc_ += Vector2f(max_p_dx, max_p_dy);
  }

  prev_landmarks_initialized = true;

  // reset the lookup table
  for (size_t i = 0; i < L_HEIGHT; ++i) {
    for (size_t j = 0; j < L_WIDTH; ++j) {
      prev_prob_landmarks[i][j] = log(k_EPSILON);
    }
  }

  // fill the lookup table using current observations
  float step_size = (angle_max - angle_min) / ranges.size();
  for (size_t i = 0; i < ranges.size(); i++) {
    float angle_i = angle_min + i * step_size;
    float range_i = ranges[i];
    if ( range_i > HORIZON - k_EPSILON  || range_i < range_min) {continue;}
    
    // get landmark position in laser frame
    float lx = range_i * cos(angle_i);
    float ly = range_i * sin(angle_i);
    
    int idx_x = (int) round((lx + HORIZON) / L_STEP);
    int idx_y = (int) round((ly + HORIZON) / L_STEP);

    int min_idx_x = idx_x - MASK_SIZE / 2;
    min_idx_x = min_idx_x < 0 ? 0 : min_idx_x;
    int max_idx_x = idx_x + MASK_SIZE / 2;
    max_idx_x = max_idx_x > (int) L_HEIGHT - 1 ? (int) L_HEIGHT - 1 : max_idx_x;
    int min_idx_y = idx_y - MASK_SIZE / 2;
    min_idx_y = min_idx_y < 0 ? 0 : min_idx_y;
    int max_idx_y = idx_y + MASK_SIZE / 2;
    max_idx_y = max_idx_y > (int) L_WIDTH - 1 ? (int) L_WIDTH - 1 : max_idx_y;
    
    // fill in range around this laser reading
    for (int x = min_idx_x; x <= max_idx_x; x+=L_STEP) {
      for (int y = min_idx_y; y <= max_idx_y; y+=L_STEP) {
        float p = prob_sensor[x - (idx_x - MASK_SIZE / 2)][y - (idx_y - MASK_SIZE / 2)];
        prev_prob_landmarks[x][y] = p > prev_prob_landmarks[x][y] ? p : prev_prob_landmarks[x][y];
      }
    }

    // add the laser scan to the map, after transforming back to the initial pose
    Vector2f kLandmark = Vector2f(lx, ly) + kLaserLoc;
    Rotation2Df r(-prev_pose_angle_);
    Vector2f mLandmark = r * kLandmark - prev_pose_loc_;
    map_.push_back(mLandmark);
  }
  
}

void SLAM::ObserveOdometry(const Vector2f& odom_loc, const float odom_angle) {
  if (!odom_initialized_) {
    prev_pose_angle_ = odom_angle;
    prev_pose_loc_ = odom_loc;
    odom_initialized_ = true;
    return;
  }
  // Keep track of odometry to estimate how far the robot has moved between 
  // poses.
  cur_odom_angle_ = odom_angle;
  cur_odom_loc_ = odom_loc;
}

vector<Vector2f> SLAM::GetMap() {
  
  // Reconstruct the map as a single aligned point cloud from all saved poses
  // and their respective scans.
  // cout << map_.size() << endl;
  return map_;
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
