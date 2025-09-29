//
// Created by Schulze on 30.01.25.
//
#include <franka_example_controllers/feedforward_joint_pd_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

namespace franka_example_controllers {

    bool FeedforwardJointPDController::init(hardware_interface::RobotHW* robot_hw,
                                      ros::NodeHandle& node_handle) {
        std::vector<std::string> joint_names;
        std::string arm_id;
        if (!node_handle.getParam("arm_id", arm_id)) {
            ROS_ERROR("FeedforwardJointPDController: Could not read parameter arm_id");
            return false;
        }
        if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
            ROS_ERROR(
                    "FeedforwardJointPDController: Invalid or no joint_names parameters provided, aborting "
                    "controller init!");
            return false;
        }

        auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
        if (model_interface == nullptr) {
            ROS_ERROR_STREAM("FeedforwardJointPDController: Error getting model interface from hardware");
            return false;
        }
        try {
            model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
                    model_interface->getHandle(arm_id + "_model"));
        } catch (hardware_interface::HardwareInterfaceException& ex) {
            ROS_ERROR_STREAM(
                    "FeedforwardJointPDController: Exception getting model handle from interface: " << ex.what());
            return false;
        }

        auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
        if (state_interface == nullptr) {
            ROS_ERROR_STREAM("FeedforwardJointPDController: Error getting state interface from hardware");
            return false;
        }
        try {
            state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
                    state_interface->getHandle(arm_id + "_robot"));
        } catch (hardware_interface::HardwareInterfaceException& ex) {
            ROS_ERROR_STREAM(
                    "FeedforwardJointPDController: Exception getting state handle from interface: " << ex.what());
            return false;
        }

        auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
        if (effort_joint_interface == nullptr) {
            ROS_ERROR_STREAM("FeedforwardJointPDController: Error getting effort joint interface from hardware");
            return false;
        }
        for (size_t i = 0; i < n_joints; ++i) {
            try {
                joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
            } catch (const hardware_interface::HardwareInterfaceException& ex) {
                ROS_ERROR_STREAM("FeedforwardJointPDController: Exception getting joint handles: " << ex.what());
                return false;
            }
        }

        std::vector<double> input_p_gains;
        if (!node_handle.getParam("k_gains", input_p_gains) || input_p_gains.size() != 7) {
            ROS_ERROR("FeedforwardJointPDController:  Invalid or no k_gain parameters provided, aborting controller init!");
            return false;
        }
        std::vector<double> input_d_gains;
        if (!node_handle.getParam("d_gains", input_d_gains) || input_d_gains.size() != 7) {
            ROS_ERROR("FeedforwardJointPDController:  Invalid or no d_gain parameters provided, aborting controller init!");
            return false;
        }

        for (uint32_t i = 0; i < 7; ++i) {
            p_gains(i) = input_p_gains[i];
            d_gains(i) = input_d_gains[i];
        }
        des_acc.setZero();
        acc_start_vel.setZero();
        cmd_start_time = ros::Time::now();
        acc_dur = ros::Duration(0.);

        // Set initial torque as the initial gravity
        std::array<double, 7> gravity_array = model_handle_->getGravity();
        Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_gravity(gravity_array.data());

        // Set the initial joint positions by reading the interface
        franka::RobotState robot_state = state_handle_->getRobotState();
        for(uint32_t i = 0; i < 7; ++i) {
            acc_start_pos(i) = robot_state.q[i];
            des_torque(i) = tau_gravity[i];
            des_pos(i) = robot_state.q[i];
            des_vel(i) = robot_state.q_d[i];
        }

        ROS_INFO_STREAM("P-GAINS: " << p_gains(0) << ", " << p_gains(1) << ", " << p_gains(2) << ", " << p_gains(3) << ", " << p_gains(4) << ", " << p_gains(5) << ", " << p_gains(6));
        ROS_INFO_STREAM("D-GAINS: " << d_gains(0) << ", " << d_gains(1) << ", " << d_gains(2) << ", " << d_gains(3) << ", " << d_gains(4) << ", " << d_gains(5) << ", " << d_gains(6));

        if (!node_handle.getParam("publish_rate", publish_rate_)){
            publish_rate_ = 500;
        }
        pub_time_initialized_ = false;
        filter_Ts = 1.0 / 1000.0;
        if (!node_handle.getParam("acc_freq_cutoff", acc_freq_cutoff)){
            acc_freq_cutoff = 500;
        }
        if (!node_handle.getParam("vel_freq_cutoff", vel_freq_cutoff)){
            vel_freq_cutoff = 500;
        }
        if (!node_handle.getParam("torque_freq_cutoff", torque_freq_cutoff)){
            torque_freq_cutoff = 100;
        }
        vel_alpha = filter_Ts /(filter_Ts + 1.0 / (2 * M_PI * vel_freq_cutoff));
        acc_alpha = filter_Ts /(filter_Ts + 1.0 / (2 * M_PI * acc_freq_cutoff));
        torque_alpha = filter_Ts /(filter_Ts + 1.0 / (2 * M_PI * torque_freq_cutoff));
        ROS_INFO_STREAM("Vel 1st Low Pass Filter | Frequency: " << vel_freq_cutoff << " Hz | alpha " << vel_alpha);
        ROS_INFO_STREAM("Acc 1st Low Pass Filter | Frequency: " << acc_freq_cutoff << " Hz | alpha " << acc_alpha);
        ROS_INFO_STREAM("Torque 1st Low Pass Filter | Frequency: " << torque_freq_cutoff << " Hz | alpha " << torque_alpha);

    
        // realtime publisher
        realtime_state_pub_.reset(new realtime_tools::RealtimePublisher<JointStateFiltered>(node_handle, "filtered_joint_states", 4));
        // get joints and allocate message
        for (unsigned i=0; i<n_joints; i++){
            realtime_state_pub_->msg_.name.push_back(joint_names[i]);
            realtime_state_pub_->msg_.position_raw.push_back(0.0);
            realtime_state_pub_->msg_.position.push_back(0.0);
            realtime_state_pub_->msg_.velocity_raw.push_back(0.0);
            realtime_state_pub_->msg_.velocity.push_back(0.0);
            realtime_state_pub_->msg_.acceleration_raw.push_back(0.0);
            realtime_state_pub_->msg_.acceleration.push_back(0.0);
            realtime_state_pub_->msg_.effort_raw.push_back(0.0);
            realtime_state_pub_->msg_.effort.push_back(0.0);
        }

        // Finally, we require a callback to get the torque command
        ROS_INFO("FeedforwardJointPDController: Waiting for torque targets on 'franka_ffwd_pd_controller/command'");
        cmd_subscriber = node_handle.subscribe("/franka_ffwd_pd_controller/command", 1,
                                                &FeedforwardJointPDController::set_command, this,
                                                ros::TransportHints().reliable().tcpNoDelay());


        return true;
    }

    void FeedforwardJointPDController::starting(const ros::Time& time) {
        des_acc.setZero();
        acc_start_vel.setZero();
        cmd_start_time = ros::Time::now();
        acc_dur = ros::Duration(0.);

        // Set the initial joint positions by reading the interface
        franka::RobotState robot_state = state_handle_->getRobotState();
        for(uint32_t i = 0; i < 7; ++i) {
            acc_start_pos(i) = robot_state.q[i];
        }

        // initialize time exactly once, to maintain publish rate through controller resets
        if (!pub_time_initialized_){
            try {
                last_publish_time_ = time - ros::Duration(1.001/publish_rate_); //ensure publish on first cycle
            } catch(std::runtime_error& ex) { // negative ros::Time is not allowed
                last_publish_time_ = ros::Time::MIN;
            }
            pub_time_initialized_ = true;
        }
        // Start filter data
        vel_old.setZero();
        acc_filtered_old.setZero();

        ROS_INFO_STREAM("INIT_POS: " << acc_start_pos(0) << ", " << acc_start_pos(1) << ", " << acc_start_pos(2) << ", " << acc_start_pos(3) << ", " << acc_start_pos(4) << ", " << acc_start_pos(5) << ", " << acc_start_pos(6));
        ROS_INFO_STREAM("INIT_VEL: " << acc_start_vel(0) << ", " << acc_start_vel(1) << ", " << acc_start_vel(2) << ", " << acc_start_vel(3) << ", " << acc_start_vel(4) << ", " << acc_start_vel(5) << ", " << acc_start_vel(6));
    }

    void FeedforwardJointPDController::set_command(const FFWDTorquePDUpdate& update) {
        // // We get the current position and velocity
        cmd_start_time = ros::Time::now();

        // Update the desired values
        for (uint32_t i = 0; i < 7; i++) {
            des_torque(i) = update.torque[i];
            des_pos(i) = update.pos[i];
            des_vel(i) = update.vel[i];
        }

        // ROS_INFO("FeedforwardJointPDController: Received data");
    }

    void FeedforwardJointPDController::update(const ros::Time& time, const ros::Duration& /*period*/) {
        // We need to discount the current gravity since gravity (and joint friction) are already compensated.
        std::array<double, 7> gravity_array = model_handle_->getGravity();
        Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_gravity(gravity_array.data());

        // Get current state
        franka::RobotState robot_state = state_handle_->getRobotState();
        Eigen::Map<Eigen::Matrix<double, 7, 1>> joint_pos(robot_state.q.data());
        Eigen::Map<Eigen::Matrix<double, 7, 1>> joint_vel(robot_state.dq.data());
        Eigen::Map<Eigen::Matrix<double, 7, 1>> joint_torque_read(robot_state.tau_J.data());

        double_t elapsed_time = (time - cmd_start_time).toSec();

        // The torque command is a combination of the feedforward torque component and a PD tracking controller
        Eigen::Matrix<double, 7, 1> pos_err = des_pos - joint_pos;
        Eigen::Matrix<double, 7, 1> vel_err = des_vel - joint_vel;
        Eigen::Matrix<double, 7, 1> tau_ff = des_torque - tau_gravity;
        Eigen::Matrix<double, 7, 1> tau_cmd = tau_ff + (p_gains * pos_err.array()).matrix() + (d_gains * vel_err.array()).matrix();

        if (elapsed_time < 0.05) {
            // ROS_INFO_STREAM("DELTA T: " << elapsed_time);
            // ROS_INFO_STREAM("DES_POS: " << des_pos(0) << ", " << des_pos(1) << ", " << des_pos(2) << ", " << des_pos(3) << ", " << des_pos(4) << ", " << des_pos(5) << ", " << des_pos(6));
            // ROS_INFO_STREAM("POS_ERR: " << pos_err(0) << ", " << pos_err(1) << ", " << pos_err(2) << ", " << pos_err(3) << ", " << pos_err(4) << ", " << pos_err(5) << ", " << pos_err(6));
            // ROS_INFO_STREAM("DES_VEL: " << des_vel(0) << ", " << des_vel(1) << ", " << des_vel(2) << ", " << des_vel(3) << ", " << des_vel(4) << ", " << des_vel(5) << ", " << des_vel(6));
            // ROS_INFO_STREAM("VEL_ERR: " << vel_err(0) << ", " << vel_err(1) << ", " << vel_err(2) << ", " << vel_err(3) << ", " << vel_err(4) << ", " << vel_err(5) << ", " << vel_err(6));
            // ROS_INFO_STREAM("TAU_FF: " << tau_ff(0) << ", " << tau_ff(1) << ", " << tau_ff(2) << ", " << tau_ff(3) << ", " << tau_ff(4) << ", " << tau_ff(5) << ", " << tau_ff(6));
        }

        for (size_t i = 0; i < 7; ++i) {
            joint_handles_[i].setCommand(tau_cmd(i));
        }

        // Update Velocity Filter
        vel_raw = joint_vel;

        for (size_t i = 0; i < 7; ++i) {
            vel_filtered[i] = first_order_low_pass_filter(vel_raw[i], vel_filtered_old[i], vel_alpha);
        }
        // Update old data
        vel_filtered_old = vel_filtered;

        // Update acceleration estimation
        // Get acc diff
        // acc_raw = (joint_vel - vel_old) / filter_Ts;
        acc_raw = (vel_filtered - vel_old) / filter_Ts;

        for (size_t i = 0; i < 7; ++i) {
            acc_filtered[i] = first_order_low_pass_filter(acc_raw[i], acc_filtered_old[i], acc_alpha);
        }
        // Update old data
        // vel_old = joint_vel;
        vel_old = vel_filtered;
        acc_filtered_old = acc_filtered;

        // Update Torque Filter
        torque_read_raw = joint_torque_read;

        for (size_t i = 0; i < 7; ++i) {
            torque_read_filtered[i] = first_order_low_pass_filter(torque_read_raw[i], torque_read_filtered_old[i], torque_alpha);
        }
        // Update old data
        torque_read_filtered_old = torque_read_filtered;

        // Publish estimation
        if (publish_rate_ > 0.0 && last_publish_time_ + ros::Duration(1.0/publish_rate_) < time){
            if (realtime_state_pub_->trylock()){
                // we're actually publishing, so increment time
                last_publish_time_ = last_publish_time_ + ros::Duration(1.0/publish_rate_);

                // populate joint state message:
                // - fill only joints that are present in the JointStateInterface, i.e. indices [0, n_joints)
                // - leave unchanged extra joints, which have static values, i.e. indices from n_joints onwards
                realtime_state_pub_->msg_.header.stamp = time;
                for (unsigned i=0; i<n_joints; i++){
                    realtime_state_pub_->msg_.position_raw[i] = joint_pos[i];
                    realtime_state_pub_->msg_.position[i] = joint_pos[i];

                    realtime_state_pub_->msg_.velocity_raw[i] = joint_vel[i];
                    realtime_state_pub_->msg_.velocity[i] = vel_filtered[i];

                    realtime_state_pub_->msg_.acceleration_raw[i] = acc_raw[i];
                    realtime_state_pub_->msg_.acceleration[i] = acc_filtered[i];

                    realtime_state_pub_->msg_.effort_raw[i] = joint_torque_read[i];
                    realtime_state_pub_->msg_.effort[i] = torque_read_filtered[i];
                }
                realtime_state_pub_->unlockAndPublish();
            }
        }
    }

    double FeedforwardJointPDController::first_order_low_pass_filter(double value, double old_value, double alpha){
        return alpha * value + (1 - alpha) * old_value;
    }

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::FeedforwardJointPDController, controller_interface::ControllerBase)
