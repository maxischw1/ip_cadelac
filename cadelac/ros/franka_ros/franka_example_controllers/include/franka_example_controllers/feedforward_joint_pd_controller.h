//
// Created by Schulze on 30.01.25.
//
#pragma once

#include <array>
#include <string>
#include <vector>

#include <franka_example_controllers/FFWDTorquePDUpdate.h>
#include <franka_example_controllers/JointStateFiltered.h>
#include <realtime_tools/realtime_publisher.h>
#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Core>

namespace franka_example_controllers {

    class FeedforwardJointPDController : public controller_interface::MultiInterfaceController<
            franka_hw::FrankaModelInterface,
            hardware_interface::EffortJointInterface,
            franka_hw::FrankaStateInterface> {
    public:
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
        void starting(const ros::Time&) override;
        void update(const ros::Time&, const ros::Duration& period) override;
        void set_command(const FFWDTorquePDUpdate&);
        double first_order_low_pass_filter(double, double, double);

    private:
        std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
        std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
        std::vector<hardware_interface::JointHandle> joint_handles_;

        static const int n_joints = 7;
        Eigen::Array<double, n_joints, 1> p_gains;
        Eigen::Array<double, n_joints, 1> d_gains;

        Eigen::Matrix<double, n_joints, 1> acc_start_pos;
        Eigen::Matrix<double, n_joints, 1> acc_start_vel;
        Eigen::Matrix<double, n_joints, 1> des_acc;


        Eigen::Matrix<double, n_joints, 1> des_torque;
        Eigen::Matrix<double, n_joints, 1> des_pos;
        Eigen::Matrix<double, n_joints, 1> des_vel;

        ros::Time cmd_start_time;
        ros::Duration acc_dur;

        ros::Subscriber cmd_subscriber;

        // State Filtering
        ros::Time last_publish_time_;
        double publish_rate_;
        bool pub_time_initialized_;
        std::shared_ptr<realtime_tools::RealtimePublisher<JointStateFiltered> > realtime_state_pub_;

        Eigen::Matrix<double, n_joints, 1> vel_raw;
        Eigen::Matrix<double, n_joints, 1> vel_old;
        Eigen::Matrix<double, n_joints, 1> acc_raw;
        Eigen::Matrix<double, n_joints, 1> torque_read_raw;

        double filter_Ts;
        double vel_alpha;
        double vel_freq_cutoff;
        Eigen::Matrix<double, n_joints, 1> vel_filtered_old;
        Eigen::Matrix<double, n_joints, 1> vel_filtered;
        double acc_alpha;
        double acc_freq_cutoff;
        Eigen::Matrix<double, n_joints, 1> acc_filtered_old;
        Eigen::Matrix<double, n_joints, 1> acc_filtered;
        double torque_alpha;
        double torque_freq_cutoff;
        Eigen::Matrix<double, n_joints, 1> torque_read_filtered_old;
        Eigen::Matrix<double, n_joints, 1> torque_read_filtered;

    };

}  // namespace franka_example_controllers
