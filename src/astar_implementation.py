#!/usr/bin/env python
import numpy as np
from geometry_msgs.msg import Twist
import tf
import rospy
from std_msgs.msg import Int64, String, Float64, Float64MultiArray
from nav_msgs.msg import Odometry, OccupancyGrid
from Astar_lib import *
from math import sqrt, atan2, ceil, pi

class A_star():

    def __init__(self):
        # load map

        rospy.init_node('robot_controller', anonymous=True)
        #self.sub= rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.map = np.load('resources/example_map.npy').T
        self.map[self.map>0] = 1
        self.resolution = 10.0/100
        self.height = self.map.shape[0]
        self.width = self.map.shape[1]
        self.rate = rospy.Rate(30)
        self.vel_msg = Twist()
        self.x = 0
        self.y = 0
        self.goal_x = 0
        self.goal_y = 0
        self.w_z = 0

        # Create node, publisher and subscriber


    def odometry_callback(self, scan):          #odometry subscriber
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

    def expand_map(self, mapa):
        """Make obstacles bigger in order to enable secure path planning for robot."""

        index_list = []
        height = self.height
        width = self.width
        mapa = np.array(mapa).reshape(height, width)
        for row_index, row in enumerate(mapa):
            for column_index, value in enumerate(row):
                if value > 0:
                    index_list.append((row_index, column_index))

        expand_positions = []
        for position in index_list:
            mapa.setflags(write=1)
            if position[0] > 0 and position[0] < height - 1:
                if position[1] > 0 and position[1] < width - 1:
                    row_indexes = [position[0] - 1, position[0], position[0] + 1]
                    column_indexes = [position[1] - 1, position[1], position[1] + 1]
                    for row_index in row_indexes:
                        for column_index in column_indexes:
                            expand_positions.append((row_index, column_index))

        # every index that is not on first or last row/column is expanded on all sides
        for position in expand_positions:
            mapa[position[0], position[1]] = 1

        return(mapa)

    def get_distance(self, goal_x, goal_y):
        """ Calculate euclidian distance between goal position and robot pose."""

        self.goal_x = (goal_x * self.resolution + self.resolution/2)
        self.goal_y = (goal_y * self.resolution + self.resolution/2)
        #print("Goal in get_distance function", self.goal_x, self.goal_y)
        #print(self.goal_x, self.goal_y)
        #print(self.x, self.y)
        distance = sqrt((self.goal_x - self.x)**2 + (self.goal_y - self.y)**2)
        return(distance)

    def publish_velocity(self, vel_msg, goal):
        """ Calculate velocities dependent on current and goal position. """

        # Proportional controller
        # linear_velocity in x_axis
        K_ro, K_alpha, K_beta = 0.3, 0.8, -0.15

        ro = self.get_distance(goal[0], goal[1])
        v_x = ro*K_ro

        # kinematic constrains for linear speed
        if v_x > 0.5:
            v_x = 0.5
        elif v_x< -0.5:
            v_x = 0.5

        # linear speed definition
        vel_msg.linear.x = v_x
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0

        self.alpha = -self.yaw + atan2(self.goal_y - self.y, self.goal_x - self.x)
        beta = -self.alpha - self.yaw
        self.w_z = K_alpha * self.alpha + K_beta

        # check for w_z saturation
        if self.w_z > 0.05:
            self.w_z = 0.05
        elif self.w_z < -0.05:
            self.w_z = -0.05

        if abs(self.alpha) > pi/12:
            vel_msg.linear.x = 0
        # angular velocity in the z-axis
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = self.w_z

        return(vel_msg)



    def run(self):

        self.pose_subscriber = rospy.Subscriber('/robot0/odom_drift',
                                                 Odometry,
                                                 self.odometry_callback)

        self.velocity_publisher = rospy.Publisher('/robot0/cmd_vel',
                                                   Twist,
                                                   queue_size=10)

        try:

            while self.x == 0 or self.y == 0:
                print(raw_input("Waiting for first measurement"))
                continue

            for i in range(2):
                working_matrix = self.expand_map(self.map)
                working_matrix[working_matrix > 0] = 1
                self.map = working_matrix

            start = (self.x, self.y)
            #finish = (130, 140)

            path = astar(working_matrix, (20, 20), (140, 140))
            show_path(working_matrix, path)


            path_cleared = []
            while not rospy.is_shutdown():
                path = path[::-1]
                while not path_cleared == path:
                    print(path)
                    for goal in path:
                        while self.get_distance(goal[0], goal[1]) >= 0.1:
                            cmd_vel_pub = self.publish_velocity(self.vel_msg, goal)

                            # check anguar speed, if fast, turn of linear speed
                            if abs(self.w_z) > 0.05:
                                cmd_vel_pub.linear.x = 0
                            self.velocity_publisher.publish(cmd_vel_pub)
                        print('Going to next position')
                        path_cleared.append(goal)
                        continue
                    rospy.spin()
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    astar_class = A_star()
try:
    astar_class.run()
except rospy.ROSInterruptException:
    pass
