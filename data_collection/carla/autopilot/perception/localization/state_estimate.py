import numpy as np
from .rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


################################################################################################
# One of the most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.10
var_imu_w = 0.10
var_gnss  = 0.10

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

class StateEstimate():
    def __init__(self, p=np.array([None]), q=np.array([None]), v=np.array([None]), t=0.0):
        """
        :param p: Position [m]
        :param q: Orientation [quaternion(a, b, c, d)]
        :param v: Velocity [m/s]
        :param t: Timestamps [ms]
        """
        self._p_init = p
        self._q_init = q
        self._v_init = v

        self._p = p
        self._q = q
        self._v = v
        self._t = t

        self.p_cov = np.zeros(9)

    ################################################################################################
    # Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
    # a function for it.
    ################################################################################################
    
    def measurement_update(self, sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
        # 3.1 Compute Kalman Gain
        r_cov = np.eye(3)*sensor_var
        k_gain = p_cov_check @ h_jac.T @ np.linalg.inv((h_jac @ p_cov_check @ h_jac.T) + r_cov)

        # 3.2 Compute error state
        error_state = k_gain @ (y_k - p_check)

        # 3.3 Correct predicted state
        p_hat = p_check + error_state[0:3]
        v_hat = v_check + error_state[3:6]
        q_hat = Quaternion(axis_angle=error_state[6:9]).quat_mult_left(Quaternion(*q_check))

        # 3.4 Compute corrected covariance
        p_cov_hat = (np.eye(9) - k_gain @ h_jac) @ p_cov_check

        return p_hat, v_hat, q_hat, p_cov_hat




    ################################################################################################
    # Now that everything is set up, we can start taking in the sensor data and creating estimates
    # for our state in a loop.
    ################################################################################################
    
    def updateState(self, data: dict) -> None:
        imu_f = data['imu_f']
        imu_w = data['imu_w']
        time = data['time']

        delta_t = time - self._t

        # 1. Update state with IMU inputs
        q_prev = Quaternion(*self._q) # previous orientation as a quaternion object
        q_curr = Quaternion(axis_angle=(imu_w*delta_t)) # current IMU orientation
        c_ns = q_prev.to_mat() # previous orientation as a matrix
        f_ns = (c_ns @ imu_f) + g # calculate sum of forces
        p_check = self._p + delta_t*self._v + 0.5*(delta_t**2)*f_ns
        v_check = self._v + delta_t*f_ns
        q_check = q_prev.quat_mult_left(q_curr)


        # 1.1 Linearize the motion model and compute Jacobians
        f_jac = np.eye(9) # motion model jacobian with respect to last state
        f_jac[0:3, 3:6] = np.eye(3)*delta_t
        f_jac[3:6, 6:9] = -skew_symmetric(c_ns @ imu_f)*delta_t


        # 2. Propagate uncertainty
        q_cov = np.zeros((6, 6)) # IMU noise covariance
        q_cov[0:3, 0:3] = delta_t**2 * np.eye(3)*var_imu_f
        q_cov[3:6, 3:6] = delta_t**2 * np.eye(3)*var_imu_w
        p_cov_check = f_jac @ self.p_cov @ f_jac.T + l_jac @ q_cov @ l_jac.T


        # 3. Check availability of GNSS measurements
        if 'gnss' in data:
            gnss = data['gnss']
            p_check, v_check, q_check, p_cov_check = \
                self.measurement_update(var_gnss, p_cov_check, gnss, p_check, v_check, q_check)
    
        # Update states (save)
        self._p = p_check
        self._v = v_check
        self._q = q_check
        self.p_cov = p_cov_check
        self._t = time
