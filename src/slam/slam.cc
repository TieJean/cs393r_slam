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
#include <fstream>
#include <string> 
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
// CONFIG_FLOAT(D_SHORT, "D_SHORT");
CONFIG_FLOAT(D_LONG, "D_LONG");
CONFIG_FLOAT(P_OUTSIDE_RANGE, "P_OUTSIDE_RANGE");
CONFIG_FLOAT(MOTION_DIST_K1, "MOTION_DIST_K1");
CONFIG_FLOAT(MOTION_DIST_K2, "MOTION_DIST_K2");
CONFIG_FLOAT(MOTION_A_K1, "MOTION_A_K1");
CONFIG_FLOAT(MOTION_A_K2, "MOTION_A_K2");
CONFIG_FLOAT(GAMMA, "GAMMA");
CONFIG_FLOAT(MIN_LOG_PROB, "MIN_LOG_PROB");

namespace slam {

config_reader::ConfigReader config_reader_({"config/slam.lua"});

SLAM::SLAM() :
    odom_initialized_(false), 
    tmp_pose_loc_(0, 0),
    tmp_pose_angle_(0),
    prev_pose_loc_(0, 0),
    prev_pose_angle_(0),
    prev_odom_loc_(0, 0),
    prev_odom_angle_(0),
    cur_odom_loc_(0, 0),
    cur_odom_angle_(0),
    prev_landmarks_initialized(false) {
      // allocates space for all three lookup tables
      prev_prob_landmarks = new float*[L_HEIGHT];
      for (size_t i = 0; i < L_HEIGHT; ++i) {
        prev_prob_landmarks[i] = new float[L_WIDTH];
      }
      prob_sensor = new float*[MASK_SIZE];
      for (size_t i = 0; i < MASK_SIZE; ++i) {
        prob_sensor[i] = new float[MASK_SIZE];
      }
      n_log = 0;
      n_lloc = 0;
      n_transformed_lloc = 0;
    }

SLAM::~SLAM() {
  for (size_t i = 0; i < L_HEIGHT; ++i) {
    delete[] prev_prob_landmarks[i];
  }
  delete[] prev_prob_landmarks;

  for (size_t i = 0; i < MASK_SIZE; ++i) {
    delete[] prob_sensor[i];
  }
  delete[] prob_sensor;

}

// initializes the motion model and observation likelihood tables
void SLAM::init() {  
  // populate the observation likelihood mask table
  size_t idx_mean_x = MASK_SIZE / 2;
  size_t idx_mean_y = MASK_SIZE / 2;
  for (size_t i = 0; i < MASK_SIZE; ++i) {
    for (size_t j = 0; j < MASK_SIZE; ++j) {
      float x = sqrt( pow(((int)idx_mean_x - (int)i) * L_STEP, 2) + pow(((int)idx_mean_y - (int)j) * L_STEP, 2) );
      if (x > CONFIG_D_LONG) {
        prob_sensor[i][j] = - pow(CONFIG_D_LONG, 2) / pow(CONFIG_SENSOR_STD_DEV, 2);
      } else {
        prob_sensor[i][j] = - pow(x, 2) / pow(CONFIG_SENSOR_STD_DEV, 2);
      }
    }
  }
}

void SLAM::GetPose(Eigen::Vector2f* loc, float* angle) const {
  // Return the latest pose estimate of the robot.
  *loc = prev_pose_loc_;
  *angle = prev_pose_angle_;
}

float getDist(const Vector2f& odom, const Vector2f& prev_odom) {
  return sqrt( pow(prev_odom.x() - odom.x(), 2) + pow(prev_odom.y() - odom.y(), 2) );
}

/**
 * curr_pose_loc: curr pose loc in odom frame
 * curr_pose_angle: curr pose angle in odom frame
 * prev_pose_loc: prev pose loc in odom frame
 * prev_pose_angle: prev pose angle in odom frame
 * lloc_curr: landmark loc in curr pose baselink/laser frame
 * @return lloc_prev: landmark loc in prev pose baselink/laser frame
 */
Vector2f transformCurrToPrev(const Vector2f& curr_pose_loc, 
                             const float& curr_pose_angle, 
                             const Vector2f& prev_pose_loc, 
                             const float& prev_pose_angle, 
                             const Vector2f& lloc_curr) {
  Rotation2Df r(curr_pose_angle - prev_pose_angle);
  return r * lloc_curr + Rotation2Df(-prev_pose_angle) * (curr_pose_loc - prev_pose_loc);
}

/**
 * @return the difference between two angles
 */
float subtractAngles(const float& a1, const float& a2) {
  float a = a1 - a2;
  a = a >  M_PI ? a - 2 * M_PI : a;
  a = a < -M_PI ? a + 2 * M_PI : a;
  return a;
}

/**
 * p(x_{i+1}|x_i, u_i)
 */
float SLAM::GetMotionLikelihood(float x, float y, float a) {
  // assume the dimensions are independent
  float d_x_mean = cur_odom_loc_.x() - prev_odom_loc_.x();
  float d_y_mean = cur_odom_loc_.y() - prev_odom_loc_.y();
  float d_a_mean = subtractAngles(cur_odom_angle_, prev_odom_angle_);

  float d_d = sqrt(pow(d_x_mean, 2) + pow(d_y_mean, 2));
  float d_stddev = CONFIG_MOTION_DIST_K1 * d_d + CONFIG_MOTION_DIST_K2 * abs(d_a_mean) ; // TODO: fix me
  float a_stddev = CONFIG_MOTION_A_K1 * d_d + CONFIG_MOTION_A_K2 * abs(d_a_mean);

  float log_x = - pow(x, 2) / pow(d_stddev, 2);
  float log_y = - pow(y, 2) / pow(d_stddev, 2);
  float log_a = - pow(a, 2) / pow(a_stddev, 2);

  return log_x + log_y + log_a;
}

/**
 * p(s_{i+1}|x_i, x_{i+1}, s_i)
 */
float SLAM::GetObservationLikelihood(const vector<float>& ranges,
                                     float range_min,
                                     float range_max,
                                     float angle_min,
                                     float angle_max,
                                     float noise_x,
                                     float noise_y,
                                     float noise_a) {
  // check all observations by using the lookup table from prev pose
  float p_landmark = 0.0;
  float step_size = (angle_max - angle_min) / ranges.size();
  
  // if ( abs(noise_a) < k_EPSILON && abs(noise_x) < k_EPSILON && abs(noise_y) < k_EPSILON ){
  //   float predicted_test_angle = subtractAngles(subtractAngles(test_angle_, -M_PI_2), M_PI_2);
  //   Vector2f test_lloc(1.0, 0.0);
  //   Vector2f predicted_curr_test_loc = transformCurrToPrev(test_loc_+Vector2f(0.5,0.5), test_angle_+M_PI_2, Vector2f(0.0, 0.0), 0.0, Vector2f(0.5, 0.5));
  //   Vector2f transformed_test_lloc = transformCurrToPrev(predicted_curr_test_loc, predicted_test_angle, test_loc_, test_angle_, test_lloc);
  //   printf("predicted curr frame: %.2f, %.2f, %.2f, ", predicted_curr_test_loc.x(), predicted_curr_test_loc.y(), predicted_test_angle);
  //   printf("cur frame: %.2f, %.2f, %.2f; prev frame: %.2f, %.2f, %.2f\n",
  //       (test_loc_+Vector2f(0.5,0.5)).x(), (test_loc_+Vector2f(0.5,0.5)).y(), test_angle_+M_PI_2,
  //        test_loc_.x(), test_loc_.y(), test_angle_);
  //   printf("test_lloc: %.2f, %.2f; transformed_test_lloc: %.2f, %.2f\n", 
  //       test_lloc.x(), test_lloc.y(), transformed_test_lloc.x(), transformed_test_lloc.y());
  //   test_loc_ += Vector2f(0.5,0.5);
  //   test_angle_ += M_PI_2;
  // }
  
  // ofstream fd_pose;
  // if(n_log == 6) {
  //   fd_pose.open("./logs/pose/" + std::to_string(n_log) + std::to_string(noise_x) + std::to_string(noise_y) + std::to_string(noise_a) + ".csv");
  //   if(!fd_pose.is_open()) perror("error opening file fd_pose!");
  // }
  for (size_t i = 0; i < ranges.size(); i++) {
    float angle_i = angle_min + i * step_size;
    float range_i = ranges[i];
    if ( range_i > HORIZON - k_EPSILON  || range_i < range_min) {continue;}

    // get landmark position in the new laser frame
    Vector2f lloc(range_i * cos(angle_i), range_i * sin(angle_i));

    float predicted_cur_pose_angle = subtractAngles(cur_odom_angle_, -noise_a);
    Vector2f predicted_cur_pose_loc = transformCurrToPrev(cur_odom_loc_, cur_odom_angle_,
                                         Vector2f(0, 0), 0.0, Vector2f(noise_x, noise_y));
    Vector2f transformed_lloc = transformCurrToPrev(predicted_cur_pose_loc, predicted_cur_pose_angle,
                                                    prev_odom_loc_, prev_odom_angle_, lloc);
    
    // get individual landmark likelihood from the lookup table
    float transformed_dist = sqrt(pow(transformed_lloc.x(), 2) + pow(transformed_lloc.y(), 2));
    if (transformed_dist > HORIZON - k_EPSILON) {continue;}
    
    size_t landmark_idx_x = (size_t) round((transformed_lloc.x() + HORIZON) / L_STEP);
    size_t landmark_idx_y = (size_t) round((transformed_lloc.y() + HORIZON) / L_STEP);
    
    p_landmark += prev_prob_landmarks[landmark_idx_x][landmark_idx_y];
    // if(n_log == 6)
    //   fd_pose << landmark_idx_x << "," << landmark_idx_y << "," << prev_prob_landmarks[landmark_idx_x][landmark_idx_y] << endl;
  }
  // if(n_log == 6) {
  //   fd_pose << p_landmark << endl;
  //   fd_pose.close();
  // }
  return p_landmark;
}

void SLAM::UpdatePose(const vector<float>& ranges,
                      float range_min,
                      float range_max,
                      float angle_min,
                      float angle_max) {
  if (!prev_landmarks_initialized) {
    // getting first pose
    map_.clear();
    prev_pose_angle_ = cur_odom_angle_;
    prev_pose_loc_ = cur_odom_loc_;
    prev_odom_angle_ = cur_odom_angle_;
    prev_odom_loc_ = cur_odom_loc_;
    return;
  }
  
  float max_p = CONFIG_MIN_LOG_PROB * 10000;
  float max_p_dx = 0.0;
  float max_p_dy = 0.0;
  float max_p_da = 0.0;
  // float max_motion = CONFIG_MIN_LOG_PROB * 10;
  // float max_observ = CONFIG_MIN_LOG_PROB * 10;
  
  // check all possible poses of the car
  for (float noise_x = -NOISE_X_BOUND; noise_x < NOISE_X_BOUND; noise_x += NOISE_D_STEP) {
    for (float noise_y = -NOISE_Y_BOUND; noise_y < NOISE_Y_BOUND; noise_y += NOISE_D_STEP) {
      for (float noise_a = -NOISE_A_BOUND; noise_a < NOISE_A_BOUND; noise_a += NOISE_A_STEP) {
        // float p_motion = GetMotionLikelihood(noise_x, noise_y, noise_a);
        float p_landmark = GetObservationLikelihood(ranges, range_min, range_max, angle_min,
                                                    angle_max, noise_x, noise_y, noise_a);
        // float prob = p_motion + CONFIG_GAMMA * p_landmark;
        // float prob = CONFIG_GAMMA * p_motion + p_landmark;
        float prob = p_landmark;
        if (prob > max_p) {
          max_p = prob;
          max_p_dx = noise_x;
          max_p_dy = noise_y;
          max_p_da = noise_a;
          // max_motion = p_motion;
          // max_observ = p_landmark;
        }
      }
    }
  }
  // cout << "max_p: " << max_p << ", " << CONFIG_GAMMA * max_motion << ", " << max_observ << endl;
  cout << "max_p_dx: " << max_p_dx << ", max_p_dy: " << max_p_dy << ", max_p_da: " << max_p_da << endl;
  tmp_pose_angle_ = prev_pose_angle_;
  tmp_pose_loc_ = prev_pose_loc_;

  Vector2f diff = transformCurrToPrev(cur_odom_loc_, cur_odom_angle_, prev_odom_loc_, prev_odom_angle_, Vector2f(max_p_dx, max_p_dy));
  Rotation2Df r(prev_pose_angle_);
  prev_pose_loc_ = prev_pose_loc_ + r * diff;
  prev_pose_angle_ = subtractAngles(subtractAngles(prev_pose_angle_, -subtractAngles(cur_odom_angle_, prev_odom_angle_)), -max_p_da);
  // prev_pose_loc_ = prev_pose_loc_ + transformCurrToPrev(cur_odom_loc_, cur_odom_angle_,
  //                                                       Vector2f(0, 0), 0.0, Vector2f(max_p_dx, max_p_dy)) - prev_odom_loc_;
  prev_odom_angle_ = cur_odom_angle_;
  prev_odom_loc_ = cur_odom_loc_;
  // cout << "n_lloc: " << n_lloc;
}

void SLAM::ReconstructMap(float lx, float ly) {
  Vector2f kLandmark = Vector2f(lx, ly) + kLaserLoc;
  Rotation2Df r(prev_pose_angle_);
  Vector2f mLandmark = r * kLandmark + prev_pose_loc_;
  map_.push_back(mLandmark);
}

void SLAM::UpdateLookupTable(const vector<float>& ranges,
                             float range_min,
                             float range_max,
                             float angle_min,
                             float angle_max) {
  // if(n_log == 6) {
  //   ofstream fd("./logs/lookup/" + std::to_string(n_log) + ".csv");
  //   // fd.open("../logs/" + std::to_string(n_log) + ".csv", ios::out);
  //   if (fd.is_open()) {
  //     for (size_t i = 0; i < L_HEIGHT; ++i) {
  //       for (size_t j = 0; j < L_WIDTH; ++j) {
  //         fd << prev_prob_landmarks[i][j] << ",";
  //         // printf("%4f, ", prev_prob_landmarks[i][j]);
  //       }
  //       fd << endl;
  //     } 
  //   } else {
  //     perror("error opening file!");
  //   }
  //   fd.close();
  // }
  // ++n_log;

  // reset the lookup table
  for (size_t i = 0; i < L_HEIGHT; ++i) {
    for (size_t j = 0; j < L_WIDTH; ++j) {
      prev_prob_landmarks[i][j] = CONFIG_MIN_LOG_PROB;
    }
  }

  // fill the lookup table using current observations
  float step_size = (angle_max - angle_min) / ranges.size();
  size_t counter = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    float angle_i = angle_min + i * step_size;
    float range_i = ranges[i];
    if (range_i > HORIZON - k_EPSILON  || range_i < range_min) {continue;}
    
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
    for (int x = min_idx_x; x <= max_idx_x; ++x) {
      for (int y = min_idx_y; y <= max_idx_y; ++y) {
        float p = prob_sensor[x - min_idx_x][y - min_idx_y];
        prev_prob_landmarks[x][y] = p > prev_prob_landmarks[x][y] ? p : prev_prob_landmarks[x][y];
      }
    }

    // add the laser scan to the map, after transforming back to the initial pose
    if (counter % DOWNSAMPLE_RATE == 0) {
      ReconstructMap(lx, ly);
    }
    counter++;
  }
}

void SLAM::ObserveLaser(const vector<float>& ranges,
                        float range_min,
                        float range_max,
                        float angle_min,
                        float angle_max) {
  // A new laser scan has been observed. Decide whether to add it as a pose
  // for SLAM. If decided to add, align it to the scan from the last saved pose,
  // and save both the scan and the optimized pose.
  if (!odom_initialized_) {return;}

  if (abs(subtractAngles(cur_odom_angle_, prev_odom_angle_)) < POSE_MIN_DELTA_A 
                   && getDist(cur_odom_loc_, prev_odom_loc_) < POSE_MIN_DELTA_D ) {return;}
  // float temp_x = prev_pose_loc_.x();
  // float temp_y = prev_pose_loc_.y();
  // float temp_a = prev_pose_angle_;

  UpdatePose(ranges, range_min, range_max, angle_min, angle_max);
  prev_landmarks_initialized = true;
  // ofstream fd1("./logs/lloc/" + std::to_string(n_lloc) + ".csv");
  // ofstream fd2("./logs/lloc_trans_noisy/" + std::to_string(n_lloc) + ".csv");
  // ofstream fd3("./logs/lloc_trans/"+ std::to_string(n_lloc) + ".csv");
  // ofstream fd4("./logs/pose/"+ std::to_string(n_lloc) + ".csv");
  // if (fd1.is_open() && fd2.is_open() && fd3.is_open() && fd4.is_open()) {
  //   float step_size = (angle_max - angle_min) / ranges.size();
  //   for (size_t i = 0; i < ranges.size(); i++) {
  //     float angle_i = angle_min + i * step_size;
  //     float range_i = ranges[i];
  //     if ( range_i > HORIZON - k_EPSILON  || range_i < range_min) {continue;}
  //     Vector2f lloc(range_i * cos(angle_i), range_i * sin(angle_i));
  //     Vector2f transformed_noisy_lloc = transformCurrToPrev(prev_pose_loc_, prev_pose_angle_,
  //                                                     tmp_pose_loc_, tmp_pose_angle_, lloc);
  //     Vector2f transformed_lloc = transformCurrToPrev(cur_odom_loc_, cur_odom_angle_,
  //                                                     tmp_pose_loc_, tmp_pose_angle_, lloc);
  //     fd1 << lloc.x() << "," << lloc.y() << endl;
  //     fd2 << transformed_noisy_lloc.x() << "," << transformed_noisy_lloc.y() << endl;
  //     fd3 << transformed_lloc.x() << "," << transformed_lloc.y() << endl;
  //   }
  //   fd4 << prev_pose_loc_.x() << "," << prev_pose_loc_.y() << "," << prev_pose_angle_ << endl;
  //   fd1.close();
  //   fd2.close();
  //   fd3.close();
  //   fd4.close();
  // } else {
  //   perror("cannot open file\n");
  //   exit(1);
  // }
  // ++n_lloc;
  

  UpdateLookupTable(ranges, range_min, range_max, angle_min, angle_max);
}

void SLAM::ObserveOdometry(const Vector2f& odom_loc, const float odom_angle) {
  if (!odom_initialized_) {
    init();
    init_pose_loc_ = odom_loc;
    init_pose_angle_ = odom_angle;
    prev_odom_loc_ = Vector2f(0, 0);;
    prev_odom_angle_ = 0;
    prev_pose_angle_ = 0;
    prev_pose_loc_ = Vector2f(0, 0);
    odom_initialized_ = true;
    test_loc_ = Vector2f(0, 0);
    test_angle_ = 0;
    cout << "init loc: (" << init_pose_loc_.x() << ", " << init_pose_loc_.y() << ") ";
    cout << "init angle: " << init_pose_angle_ << endl;
  }
  // Keep track of odometry to estimate how far the robot has moved between 
  // poses.
  Rotation2Df r(-init_pose_angle_);
  cur_odom_angle_ = odom_angle - init_pose_angle_;
  cur_odom_loc_ = r * (odom_loc - init_pose_loc_);
}

vector<Vector2f> SLAM::GetMap() {
  // Reconstruct the map as a single aligned point cloud from all saved poses
  // and their respective scans.
  return map_;
}

}  // namespace slam


