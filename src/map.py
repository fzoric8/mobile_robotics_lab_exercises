#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from sensor_msgs.msg import Range
from math import atan2, pi, asin, tanh, atan, sin, cos, e, ceil, sqrt, log
from std_msgs.msg import Int64, String, Float64, Float64MultiArray
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
import tf

map_width=1550    #real map dimensions in cm
map_height=1492
resolution=10.0   #grid tiles are 10x10 cm

class map_class():
    def __init__(self):
        rospy.init_node('mapping_node')
        self.occupancy_grid = OccupancyGrid()                                           #defining output occupancy grid
        self.occupancy_grid.info.resolution= resolution/100 # this is in meters
        self.occupancy_grid.info.width=int(ceil(map_width/resolution))
        self.occupancy_grid.info.height=int(ceil(map_height/resolution))
        self.occupancy_grid.data = np.full((self.occupancy_grid.info.width,
                                            self.occupancy_grid.info.height),
                                           -1, dtype=float)
        # updated_odds initiialized as numpa array
        self.updated_odds = np.zeros(shape=(1, self.occupancy_grid.info.width *
                                            self.occupancy_grid.info.height))

        self.yaw=0
        self.x=0
        self.y=0
        self.resolution = resolution

        #SIMULATION PARAMETERS#
        self.Rmax= 3              # sonar max range in m
        self.rov = 100            # visibility radius in cm
        self.th3db = 0.5          # half-width of the sensor beam in radians
        self.pE = 0.4             # lower limit of the conditional probability
        self.pO = 0.6             # upper limit of the conditional probability
        self.deltark = 10.0 / 100      # parameter which designates the area in which the sensor measurement r takes the average value

        self.sonardata=[]
        self.noised_sonardata=np.zeros(16)
        self.sonar_coordinates=[ [10, 0],   [10, 5],  [10, 10],                 #relative sonar coordinates in cm
                                 [5, 10],   [0, 10],  [-5, 10],  [-10, 10],
                                 [-10, 5],  [-10, 0], [-10, -5], [-10, -10],
                                 [-5, -10], [0, -10], [5, -10],  [10, -10],
                                 [10, -5]]
        self.sonar_thetas=[]
        for i in range(9):
            self.sonar_thetas.append(i*pi/8)                #sonar orientations in radians
        for i in range(7):
            self.sonar_thetas.append(-(7-i)*pi/8)


    def sonar_callback(self, scan):     #sonar subscriber
        self.sonardata=scan.data

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
        # Odometry scan.pose is represented in meters, i converted it to cm's in order to calculate ro
        self.x=scan.pose.pose.position.x
        self.y=scan.pose.pose.position.y

    def inverse_sonar_model(self, ro, theta, sensor_measurement):
        """Inverse_sonar_model returns probability that certain cell is occupied.

            prams:: ro --> distance from grid cell and robot
                    theta --> angle between
                    sensor_measurement  --> range from odometry scan(m)
        """

        if (ro < (sensor_measurement - 2 * self.deltark)):
            return (0.5 + (self.pE - 0.5) * self.alpha(theta) * self.delta(ro))
        elif (
              (ro < (sensor_measurement - self.deltark)) and
              (ro >= (sensor_measurement - 2 * self.deltark))
             ):
            return (0.5 + (self.pE - 0.5) * self.alpha(theta) * self.delta(ro) * (1 - (2 + ((ro - sensor_measurement)/self.deltark) ** 2)))
        elif (
              (ro >= (sensor_measurement - self.deltark)) and
              ((sensor_measurement + self.deltark) > ro)
             ):
            return (0.5 + (self.pO - 0.5) * self.alpha(theta) * self.delta(ro) * (1 - ((ro - sensor_measurement)/self.deltark) ** 2))
        elif (ro > (sensor_measurement + self.deltark)):
            return (0.5)

    def alpha(self, theta):
        """Modulation function which represents the main bump of sensor ray."""

        if (abs(theta) >= 0 and abs(theta) <= (self.th3db)):
            return (1 - (theta/self.th3db)**2)
        elif (abs(theta) > self.th3db):
            return 0

    def delta(self, ro):
        """Radial modulation function which causes sensor reading with large
        radius to have small impact on whole measurement."""

        return (1-(1+tanh(2*(ro - self.rov/100)))/2)

    def calculate_ro(self, sorted_grid_positions, sonar_pos_x, sonar_pos_y):
        """Calculate distance between sonar and occupancy grid cell location."""
        calculated_ro_s = []
        for grid_loc in sorted_grid_positions:
            calculated_ro_s.append(sqrt((sonar_pos_x - grid_loc[0])**2 + (sonar_pos_y - grid_loc[1])**2))
        return(calculated_ro_s)

    def calculate_theta(self, sorted_grid_positions, sonar_pos_x, sonar_pos_y, i):
        """Calculate theta (angle between ro and x axis of a occupancy_grid)."""
        calculated_theta_s = []
        for grid_loc in sorted_grid_positions:
            angle = atan2((grid_loc[1] - sonar_pos_y),(grid_loc[0] - sonar_pos_x))
            # determined by experimenting
            calculated_theta_s.append(-angle + self.yaw + self.sonar_thetas[i])
        return(calculated_theta_s)

    def conditional_probability(self, i):
        """This function returns calculated conditional probabiity of all the cells affected
           by the single sonar measurment according to the inverse sensor model.

           params:
           ro --> distance between sonar and cell(i,j)
           theta --> angle between sonar ray and robot angle from cell(i,j)
        """

        # distance between sonar and robot in meters
        sonar_robot_distance = sqrt(self.sonar_coordinates[i][0]**2 + self.sonar_coordinates[i][1]**2)/100

        # position of sonar in x, y coordinates
        robot_sonar_angle = self.yaw + self.sonar_thetas[i]
        self.sonar_x = self.x + sonar_robot_distance*np.sin(robot_sonar_angle)
        self.sonar_y = self.y + sonar_robot_distance*np.cos(robot_sonar_angle)

        #resolution of occupancy_grid
        resolution = self.occupancy_grid.info.resolution
        grid_positions = []

        # calculate grid positions for each cell in grid
        for grid_y in range(self.occupancy_grid.info.height):
            for grid_x in range(self.occupancy_grid.info.width):
                grid_positions.append((grid_x * resolution + resolution/2,
                                  grid_y * resolution + resolution/2))

        # distances between each cell and sensor position
        ro_list = self.calculate_ro(grid_positions, self.sonar_x, self.sonar_y)

        # sensor_angle --> angle between sensor and ro between each cell
        theta_list = self.calculate_theta(grid_positions, self.sonar_x, self.sonar_y, i)

        # calulate probabilities with inverse_sonar_model
        cond_prob_list = []
        for ro_, theta_ in zip(ro_list, theta_list):
            if ro_ > self.rov/100:
                cond_prob_list.append(0.5)
            else:
                cond_prob_list.append((self.inverse_sonar_model(ro_, theta_, self.sonardata[i])))

        # function returns list with conditional probabilities
        return np.array(cond_prob_list)

    def do_mapping(self):
        """Recursive cell occupancy calculation.

        A new occupancy probability will be calculated for the cells which were
        affected by the measurement i and update the values of the occupancy grid map"""

        # recursive cell occupancy calculation
        for i in range(len(self.sonardata)):

            # calculate conditional probability for each cell and every sensor measurement
            conditional_probabilities = self.conditional_probability(i)

            # calculate odds for calculated conditional probabilties
            new_odds = self.calculate_odds(conditional_probabilities)

            # update old_odds with new calculated odds
            self.updated_odds += new_odds

        # return calculated probabilities in occupancy_grid data
        self.occupancy_grid.data = self.calculate_probs(self.updated_odds[0])


    def calculate_odds(self, cond_probabilites):
        """Calculate odds with given conditional probs."""

        return(np.array((map(lambda x: np.log(x/(1-x)), cond_probabilites))))

    def calculate_probs(self, calculated_odds):
        """Calculate probs with all odds updated."""

        calculated_probs = []
        for odd in calculated_odds:
                calculated_probs.append(int((1/(1+np.exp(-odd)))*100))
        return(calculated_probs)

    def run(self):

        self.sub= rospy.Subscriber('/robot0/sonar_data', Float64MultiArray, self.sonar_callback)       #defining the subscribers and publishers
        self.sub= rospy.Subscriber('/robot0/odom', Odometry, self.odometry_callback)
        self.OG_publisher = rospy.Publisher('OG_map', OccupancyGrid, queue_size=10)     #occupancy grid publisher
        r=rospy.Rate(30)
        try:
            while not rospy.is_shutdown():
                if len(self.sonardata) == 0:
                    continue
                self.do_mapping()
                self.OG_publisher.publish(self.occupancy_grid)    #run the code and publish the occupancy grid
                r.sleep()
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    mapping = map_class()
    try:
        mapping.run()
    except rospy.ROSInterruptException:
        pass
