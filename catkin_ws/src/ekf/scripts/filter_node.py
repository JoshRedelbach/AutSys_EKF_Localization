#!/usr/bin/env python3

# --- Import necessary libraries --- 
import os
import rospy
import numpy as np
import cv2

# --- Import ROS message types ---
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler

# --- Add the parent directory to the Python path ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# -- Import custom modules ---
import ekf.EKFJ as EKFJ
import ekf.data_prepro as data_prepro


class EKFLocalizer:

    def __init__(self, mu_init, Sigma_init, map_points):
        """
        Initialize the EKF Localizer node.
        This function sets up the initial state, covariance, map points, and subscribes to the necessary topics.
        Args:
            mu_init (np.array): Initial state estimate (x, y, theta).
            Sigma_init (np.array): Initial covariance matrix.
            map_points (np.array): Map points for the EKF update.
        """
        # EKF instance
        Q = np.diag([0.001, 0.001, np.deg2rad(2)])**2
        R_icp = np.diag([0.1, 0.1, np.deg2rad(2)])**2
        R_cam = np.diag([0.1, 0.1, np.deg2rad(2)])**2

        self.ekf = EKFJ.EKF(mu_init, Sigma_init, map_points, Q=Q, R_icp=R_icp, R_cam=R_cam, tolerance_icp=50, tolerance_cam=50)

        # Helper variables
        self.last_odom = None
        self.current_pose = mu_init
        self.trajectory_log = []
        self.image_counter = 0

        # Publishers
        self.pose_pub = rospy.Publisher("/ekf_pose", PoseStamped, queue_size=1)
        self.path_pub = rospy.Publisher("/ekf_path", Path, queue_size=1)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        self.publish_pose(False)            # publish initial pose

        # Subscribers
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback)

        # Function for shutting down --> save trajectory and print statistics
        rospy.on_shutdown(self.shutdown_hook)

        # Debug print
        print("EKF LOCALIZATION NODE INITIALIZED")


    def odom_callback(self, msg):
        """ 
        Callback for odometry messages.
        This function processes the odometry data, predicts the EKF state, and publishes the updated pose.
        """
        # Convert odometry message to pose
        current_odom = data_prepro.odom_msg_to_pose(msg)
        
        # Extract velocity from the message --> required for camera update step
        velocity = msg.twist.twist.linear.x

        # Call the EKF predict step with the odometry data
        # If this is the first odometry message, only initialize the last_odom
        if self.last_odom is not None:
            u = (self.last_odom, current_odom)
            self.current_pose = self.ekf.predict(u, velocity)
        self.last_odom = current_odom

        # Publish the updated pose with flag 0 for odometry id
        self.publish_pose(0)


    def scan_callback(self, scan_msg):
        """ 
        Callback for laser scan messages.
        This function processes the laser scan data, updates the EKF state, and publishes the updated pose.
        """
        # Convert laser scan message to points
        scan = data_prepro.scan_to_points(scan_msg)

        # Check if the scan contains valid points
        if scan.size == 0:
            rospy.logwarn("No valid scan points — skipping EKF update")
            return
        
        # Run EKF update using laser scan
        self.current_pose = self.ekf.update_ICP(scan)

        # Publish the updated pose with flag 1 for laser scan id
        self.publish_pose(1)


    def image_callback(self, msg):
        """
        Callback for compressed image messages.
        This function processes the image data, updates the EKF state, and publishes the updated pose.
        """
        # Increment the image counter
        self.image_counter += 1

        # Process the image every 10th message
        # This is to reduce the frequency of image processing
        # and to avoid overloading the EKF with too many updates
        if self.image_counter == 10:
            self.image_counter = 1                                      # Reset the counter
            time = msg.header.stamp.to_sec()                            # Get the timestamp of the image   

            np_arr = np.frombuffer(msg.data, np.uint8)                  # Convert the compressed image data to a numpy array   
            gray_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)     # Decode the image to grayscale

            # Call the EKF update step witht the image
            self.current_pose = self.ekf.update_CAM(gray_image, time)

            # Publish the updated pose with flag 2 for camera id
            self.publish_pose(2)


    def publish_pose(self, flag_sensor):
        """
        Publish the current pose and update the path for RViz visualization.
        This function retrieves the current state from the EKF, constructs a PoseStamped message,
        and publishes it along with the path.
        """
        # Get the current state (mean and covariance) from the EKF
        mu, Sigma = self.ekf.get_state()

        # Create a PoseStamped message to publish the current pose
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = mu[0]
        pose.pose.position.y = mu[1]
        pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, mu[2])
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        # Publish the pose
        self.pose_pub.publish(pose)

        # Update path for RViz
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

        # Log trajectory with covariance
        self.trajectory_log.append([
            rospy.Time.now().to_sec(),
            mu[0],
            mu[1],
            mu[2],
            *Sigma.flatten(),
            flag_sensor
        ])


    def shutdown_hook(self):
        """
        Shutdown hook to save the trajectory and print statistics.
        This function is called when the ROS node is shutting down.
        It saves the trajectory to a CSV file and prints average runtimes and rejections.
        """
        rospy.loginfo("Shutting down EKF node. Saving trajectory to ekf_trajectory.csv...")
        try:
            # Save the trajectory log to a CSV file
            np.savetxt("ekf_trajectory.csv", np.array(self.trajectory_log), delimiter=",",
            header="time,x,y,theta,sigma_00,sigma_01,sigma_02,sigma_10,sigma_11,sigma_12,sigma_20,sigma_21,sigma_22,flag_sensor",
            comments='')
            rospy.loginfo("Trajectory saved.")

            # Print average runtimes and rejections
            avg_runtime_icp = self.ekf.runtime_total_updates_icp / self.ekf.number_of_updates_icp
            avg_runtime_cam = self.ekf.runtime_total_updates_cam / self.ekf.number_of_updates_cam
            print("Average runtime of a ICP iteration: ", avg_runtime_icp)
            print("Average runtime of a Cam iteration: ", avg_runtime_cam)

            print("\nNumber of rejections ICP: ", self.ekf.number_of_rejections_icp)
            print("Number of rejections CAM: ", self.ekf.number_of_rejections_cam)

        except Exception as e:
            rospy.logerr(f"Failed to save trajectory: {e}")



if __name__ == "__main__":
    """ Main function to initialize the ROS node and start the EKF localization process."""
    # Initialize the ROS node
    rospy.init_node("ekf_localization_node", log_level=rospy.INFO)

    # Load the map once
    # !! Make sure the map file paths are correct and accessible !!
    home = os.path.expanduser("~")
    pgm_file = f"{home}/Data/full_room_01062025.pgm"
    yaml_file = f"{home}/Data/full_room_01062025.yaml"
    map_points = data_prepro.load_map(pgm_file, yaml_file)
    rospy.loginfo(f"Map loaded with {len(map_points)} points")

    # Initial pose estimate and uncertainty
    mu_init = np.array([-4, 4, np.deg2rad(180)])
    Sigma_init = np.eye(3) * 10
    
    # Create an instance of the EKFLocalizer
    ekf_node = EKFLocalizer(mu_init, Sigma_init, map_points)
    
    # Keep the node running
    rospy.spin()
