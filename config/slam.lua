-- SENSOR_STD_DEV = 0.1;
-- SENSOR_STD_DEV = 0.3; -- works best
SENSOR_STD_DEV = 0.6;
D_SHORT = SENSOR_STD_DEV * 1.5;
D_LONG = SENSOR_STD_DEV * 2.0;
P_OUTSIDE_RANGE = 1e-4;

MOTION_DIST_K1 = 0.3;
MOTION_DIST_K2 = 0.05;
MOTION_A_K1 = 0.1 * 50;
MOTION_A_K2 = 1.0 * 100;

GAMMA = 0.01;

