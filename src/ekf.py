#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from sensor_msgs.msg import Range
from math import atan2, pi, asin, tanh, atan, sin, cos, e, ceil, floor, sqrt
from std_msgs.msg import Int64, String, Float64, Float64MultiArray
import numpy as np
import time
from nav_msgs.msg import Odometry, OccupancyGrid
import tf

class ekf_class():
    def __init__(self):
        rospy.init_node('ekf_node')
        self.sonar_coordinates=[ [10, 0],   [10, 5],  [10, 10],
                                 [5, 10],   [0, 10],  [-5, 10],  [-10, 10],
                                 [-10, 5],  [-10, 0], [-10, -5], [-10, -10],
                                 [-5, -10], [0, -10], [5, -10],  [10, -10],
                                 [10, -5]]
        self.sonar_thetas=[]
        for i in range(9):
            self.sonar_thetas.append(i*pi/8)
        for i in range(7):
            self.sonar_thetas.append(-(7-i)*pi/8)
        self.sonardata=[]
        self.yaw=0
        self.x=0
        self.y=0
        self.hz_rate=30
        # cm^2 --> m^2
        self.var_D = 2.0/(100**2)
        # squared degrees --> squared_radians
        self.var_delta_theta = 2 * (pi/180)**2
        # initialize  P_k and observe how changing those values affects the whole simulation
        # big mistake in starting moment
        self.P_k = np.identity(3)*10
        self.P_k.setflags(write=1)
        self.P_k[2,2] = 100
        self.wheel_velocity=Twist()
        self.height = 0
        self.width = 0
        self.resolution = 0
        self.map = []
        self.sample_T = None
        # sonar characteristics
        self.rov= 100/100            # visibility radius in cm
        self.Rmax=3
        self.th3db = 0.5          # half-width of the sensor beam in radians
        self.sensor_var = 100.0/100**2

        # Initialize corrected odometry
        self.corrected_odometry = Odometry()
        self.corrected_odometry.child_frame_id = "robot0"
        self.corrected_odometry.header.frame_id = "map_static"

        # Transform broadcaster
        self.tb = tf.TransformBroadcaster()

        self.tb.sendTransform((1, 2, 0),
                              tf.transformations.quaternion_from_euler(0, 0, 0),
                              rospy.Time.now(),
                              "robot0",  # child
                              "map_static"  # parent
                              )

    def sonar_callback(self, scan):     #sonar subscriber
        self.sonardata=scan.data

    def odometry_callback(self, scan):          #odometry subscriber
        self.wheel_velocity=scan.twist.twist
        quaternion = (
            scan.pose.pose.orientation.x,
            scan.pose.pose.orientation.y,
            scan.pose.pose.orientation.z,
            scan.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion) #transforming quaternions to euler
        #self.roll = euler[0]
        #self.pitch = euler[1]
        self.yaw = euler[2]
        self.x=scan.pose.pose.position.x
        self.y=scan.pose.pose.position.y

    def map_callback(self, scan):

        self.height = scan.info.height
        self.width = scan.info.width
        self.resolution = scan.info.resolution
        self.map = scan.data
        #print(self.resolution)


    def get_obstacles(self, map_server):
        """ Get obstacles from map topic."""

        self.obstacle_list = []
        for index, element in enumerate(map_server):
            if element > 0:
                self.obstacle_list.append(index)
        return(self.obstacle_list)


    def sonarpredict(self):
        """Calculate H matrix using this function.

        H matrix is calculated with the help of obstacle locations and a-priori
        x_state var prediction"""

        # Calculate sample time --> # finish_time - time that program entered_this_func
        if not self.sample_T:
            sample_T = 1.0/2
        else:
            self.sample_T = self.finish_time - start_time
            start_time = time.time()



        # create robot position and orientation state variable
        # initialize it only once after we get first measurement, not here
        x_state_var = np.array([[self.x], [self.y], [self.yaw]])

        # control variables (linar and angular velocity)
        u_linear_velocity_x = self.wheel_velocity.linear.x
        w_angular_velocity_theta = self.wheel_velocity.angular.z

        u_state_var = np.array([[w_angular_velocity_theta*(sample_T)],
                               [u_linear_velocity_x*(sample_T)],
                               [0]])

        # calculate neccessary error matrix prediction
        A_k = self.calculate_A(u_state_var[1], x_state_var[2], u_state_var[0])
        W_k = self.calculate_W(u_state_var[1], x_state_var[2], u_state_var[0])
        Q_k = self.calculate_Q(u_state_var[0])

        # calculate x_prior in k+1
        self.x_prior = self.predict_x_next(x_state_var, u_state_var, [0, 0, 0])

        # calculate P_prior in k+1
        self.P_prior = self.predict_P_next(A_k, self.P_k, W_k, Q_k)

        #print('nonlinear_func', self.nonlinear_function(x_state_var, u_state_var, [0, 0, 0]))

        sonar_sim_list = []
        sonar_sim_list = map(self.hit_sonar , [x for x in range(15) if x % 2])
        # sonar_sim_list_append = sonar_sim_list.append
        # for i in [x for x in range(15) if x%2]:
        #     sonar_sim_list.append(self.hit_sonar(i))

        # estimated obstacle location with good obstacles
        object_loc = [x[-1] for x in sonar_sim_list if abs(x[0]-x[1]) < 0.8]

        # sonar diffs (y(k+1) - h(x(k+1), 0)
        self.sonar_diffs = [x[1] - x[0] for x in sonar_sim_list if abs(x[0] - x[1]) < 0.8]

        #H_matrix = self.calculate_H(sonar_sim_est[0], sonar_sim_est[1])
        H_matrix = self.calculate_H(object_loc)

        # these are correct
        V_matrix = np.eye(H_matrix.shape[0])
        R_matrix = np.eye(H_matrix.shape[0])*self.sensor_var

        return(H_matrix, V_matrix, R_matrix)

    def calculate_H(self, obstacle_loc):
        """Calculate H matrix using obstacles detected by sonars."""

        H = np.empty((0,3), float)
        #print(obstacle_loc)
        for loc in obstacle_loc:
            H = np.append(H, self.calculate_row_H(loc[0], loc[1]).T, axis=0)
        return(H)

    def calculate_row_H(self, x_pi, y_pi):
        #print(x_pi, y_pi)
        row = np.array([self.x_prior[0] - x_pi, self.x_prior[1] - y_pi, [0]])
        scaling_factor = 1.0/sqrt( (self.x_prior[0] - x_pi)**2 +
                                   (self.x_prior[1] - y_pi)**2)
        return(row*scaling_factor)


    def predict_x_next(self, x_state_var, u_state_var, w):
        """Nonlinear function which gives us x_minus_k+1 = f(x_k, u_k, w_k).
        This is used for a-priori state estimation

        params:: x_state_var --> vector which contains state variables,
                                [pose_x_k, pose_y_k, theta_k]
                 u_state_var --> vector which contains control state vars,
                                [delta_theta_k, D_k, 0]
                 w_state_var --> process noise, unpossible to predict
                                [w_theta , w_d, 0]
        """

        # print(x_state_var, u_state_var, w)
        # calculate x_hat_minus_x_k+1
        x_hat_minus_x = (x_state_var[0] + (u_state_var[1] + w[1]) * np.cos(
                        x_state_var[2] + u_state_var[0] + w[0]
                        ))
        # calculate x_hat_minus_y_k+1
        x_hat_minus_y = (x_state_var[1] + (u_state_var[1] + w[1]) * np.sin(
                        x_state_var[2] + u_state_var[0] + w[0]
                        ))
        # calculate x_hat_minus_theta_k+1
        x_hat_minus_theta = x_state_var[2] + u_state_var[0] + w[0]

        # create vector from calculated x_hat_minus_k+1
        self.x_prior = np.array([x_hat_minus_x,
                                 x_hat_minus_y,
                                 x_hat_minus_theta])
        # return a-priori estimation
        # This is correct!
        return(self.x_prior)

    def calculate_A(self, D_k, theta_hat_k, delta_theta_k):
        """ Calculate A matrix which is used in linearization.

        Params are same as in nonlinear_function"""

        A = np.identity(3)
        A.setflags(write=1)
        A[0, 2] = -D_k*np.sin(theta_hat_k + delta_theta_k)
        A[1, 2] = D_k*np.cos(theta_hat_k + delta_theta_k)
        return(A)

    def calculate_W(self, D_k, theta_hat_k, delta_theta_k):
        """ Calculate W matrix which is used in linearization.

        Params are same as in nonlinear_function"""

        W = np.zeros(shape=(3,2))
        W.setflags(write=1)
        # print(D_k)
        # print(np.sin(theta_hat_k + delta_theta_k))
        # print(np.cos(theta_hat_k + delta_theta_k))
        W[0, 0] = -D_k*np.sin(theta_hat_k + delta_theta_k)
        W[0, 1] = np.cos(theta_hat_k + delta_theta_k)
        W[1, 0] = D_k*np.cos(theta_hat_k + delta_theta_k)
        W[1, 1] = np.sin(theta_hat_k + delta_theta_k)
        W[2, 0] = 1
        return(W)

    def calculate_Q(self, delta_theta_k):
        """ Calcute Q matrix which depends on delta_theta_k.

        Param is delta_theta_k because variances are constant."""

        Q = np.zeros(shape=(2, 2))
        Q.setflags(write=1)
        Q[0, 0] = delta_theta_k**2 * self.var_delta_theta**2
        Q[1, 1] = self.var_D**2
        return(Q)

    def predict_P_next(self, A_k, P_k, W_k, Q_k):
        """ Predict P matrix given calculated

        Return P_prior
        """

        P_pred_next = ( np.linalg.multi_dot([A_k, P_k, A_k.T]) +
                        np.linalg.multi_dot([W_k, Q_k, W_k.T])
                       )

        return(P_pred_next)

    def correct(self):
        """This function uses newest noised odometry data and sonarpredict function
        in order to correct a-priori prediction with sonar measurements"""

        sonarpredict_data = self.sonarpredict()
        H = sonarpredict_data[0]
        V = sonarpredict_data[1]
        R = sonarpredict_data[2]

        S = np.linalg.multi_dot([H, self.P_prior, H.T]) + np.linalg.multi_dot([V, R, V.T])
        K = np.linalg.multi_dot([self.P_prior, H.T, np.linalg.inv(S)])
        self.P_k = self.P_prior - np.linalg.multi_dot([K, S, K.T])

        self.x_hat = self.x_prior + np.dot(K, np.array(self.sonar_diffs).reshape(-1, 1))

        self.publish_corrected_odometry(self.x_hat)



    def publish_corrected_odometry(self, x_hat):
        """ Publish corrected odometry to ROS topic in order to
        visually compare actual noised and corrected odometry."""

        self.corrected_odometry.pose.pose.position.x = np.asscalar(x_hat[0])
        self.corrected_odometry.pose.pose.position.y = np.asscalar(x_hat[1])
        quaternion = tf.transformations.quaternion_from_euler(0, 0, np.asscalar(x_hat[2]))
        self.corrected_odometry.pose.pose.orientation.x = quaternion[0]
        self.corrected_odometry.pose.pose.orientation.y = quaternion[1]
        self.corrected_odometry.pose.pose.orientation.z = quaternion[2]
        self.corrected_odometry.pose.pose.orientation.w = quaternion[3]

        self.corr_publisher.publish(self.corrected_odometry)

        self.finish_time = time.time()




    def find_obstacle_loc(self, obstacle_list):
        """ Find obstacle positions with obstacle list.

        Function returns locs list which contains tuples that contains tuples
        which have (x,y) position for each obstacle that map contains."""

        x_obst = []
        y_obst = []
        #x_obst_append = x_obst.append
        #y_obst_append = y_obst.append
        locs = []

        for x in obstacle_list:
            if x < self.width:
                x_obst.append(x*self.resolution + self.resolution/2)
            else:
                x_obst.append((x % self.width)*self.resolution + self.resolution/2)

        for y in obstacle_list:
            y_obst.append((y/self.width)*self.resolution + self.resolution/2)

        locs = map(lambda x: x, zip(x_obst, y_obst))

        return(locs)

    def hit_sonar(self, i):
        """ This function returns the simulated distance to the closest obstacle
        with the help of sonar_data, object_locations and x_prior."""

        est_obstacle_dist_ = 1000
        closest_object = None

        # calculate sonar distance from the center of robot0
        sonar_distance = sqrt(self.sonar_coordinates[i][0]**2 +
                              self.sonar_coordinates[i][1]**2) / 100
        # calculate angle of robot + sonar_angle
        sonar_robot_angle = self.x_prior[2] + self.sonar_thetas[i]

        # calculate predicted sonar position
        sonar_x = self.x_prior[0] + sonar_distance*sin(sonar_robot_angle)
        sonar_y = self.x_prior[1] + sonar_distance*cos(sonar_robot_angle)

        for object_loc in self.obstacle_locs:

            dist_x = object_loc[0] - sonar_x
            dist_y = object_loc[1] - sonar_y

            # distance between robot and obstacle
            estimated_robot_object_dist = sqrt(dist_x**2 + dist_y**2)

            # if obstacle is out of sensor range compare it with other obstacle
            if estimated_robot_object_dist > self.Rmax:
                continue

            angle = atan2(dist_y, dist_x)
            theta = -angle + sonar_robot_angle

            # if obstacle is outside sensor angle check other obstacles
            if theta > self.th3db:
                continue

            dist_obst_sonar_x = self.x_prior[0] - object_loc[0]
            dist_obst_sonar_y = self.x_prior[1] - object_loc[1]

            # measurement of i-th sonar
            est_obstacle_dist = sqrt(dist_obst_sonar_x**2 + dist_obst_sonar_y**2)

            # save closest obstacle
            if est_obstacle_dist < est_obstacle_dist_:
                est_obstacle_dist_ = est_obstacle_dist
                closest_object = object_loc

            # error_hack for object_loc sensor
        if not closest_object:
            closest_object = object_loc

        # returns estimated obstacle distance, sonar measurement and nearest obstacle location
        return(est_obstacle_dist_, self.sonardata[i], closest_object)


    def run(self):

        self.sub= rospy.Subscriber('/robot0/sonar_data',Float64MultiArray, self.sonar_callback)       #defining the subscribers
        self.sub= rospy.Subscriber('/robot0/odom_drift', Odometry, self.odometry_callback)
        self.sub= rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        r=rospy.Rate(self.hz_rate)

        # Corrected odometry publisher
        self.corr_publisher = rospy.Publisher("/robot0/odom_corr", Odometry, queue_size=10)
        self.MSE_error_publisher = rospy.Publisher("/MSE", Float64)

        try:
            while len(self.map) == 0:
                continue

            self.obstacle_list = self.get_obstacles(self.map)
            self.obstacle_locs = self.find_obstacle_loc(self.obstacle_list)

            while not rospy.is_shutdown():
                self.correct()
                self.MSE_error_publisher.publish(np.trace(self.P_k))
                r.sleep()

        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    ekf = ekf_class()
    try:
        ekf.run()
    except rospy.ROSInterruptException:
        pass
