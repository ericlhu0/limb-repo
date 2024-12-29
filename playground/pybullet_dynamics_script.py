"""PyBullet Script from old Limb Repo repo."""

import time
import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import matplotlib.pyplot as plt
import scipy
import os
import sys 
from scipy.spatial.transform import Rotation as R

import argparse

from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--delta', type=float, default=1.0/200.0)
parser.add_argument('--control', type=str, default="robot_torque")
parser.add_argument('--debug', action="store_true")

p = bc.BulletClient(connection_mode=pybullet.DIRECT)

from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv

parsed_config = LRPyBulletEnv.parse_config("/home/eric/lr-dir/limb-repo/assets/configs/test_env_config.yaml")
parsed_config.pybullet_config.use_gui = True
math_dynamics = MathDynamicsNoNVector(parsed_config)

def test_control_output(cid, human_id, robot_id, delta_t=1.0/240.0, control="robot_torque", robot_to_human_ee=None): #time step of the pybullet sim

    # --- enable force torque sensors for human and panda arms --- #
    utils.enable_force_torque(robot_id)
    utils.enable_force_torque(human_id)

    # --- get starting angular positions and velocities for joints --- #
    pos_h_previous, vel_h_previous, mtorq_h, mforce_h = utils.getMotorJointStates(human_id)
    pos_r_previous, vel_r_previous, mtorq_r, mforce_r = utils.getMotorJointStates(robot_id)

    # --- set desired time step --- #
    p.setTimeStep(delta_t)

    torques = []
    acc_r_sim = []
    velocity_h = []
    velocity_r = []
    velocity_h_sim = []
    velocity_r_sim = []

    position_h = []
    position_r = []
    position_h_sim = []
    position_r_sim = []

    # --- start logging --- #
    log_id = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "logs/state_log.txt")
    if DEBUG:
        for i in utils.get_good_joints(robot_id):
            print(p.getJointInfo(robot_id, i))
        for i in utils.get_good_joints(human_id):
            print(p.getJointInfo(human_id, i))  


    mvel_h_previous = vel_h_previous
    mvel_r_previous = vel_r_previous
    mpos_h_previous = pos_h_previous
    mpos_r_previous = pos_r_previous

    mvel_h = vel_h_previous
    mvel_r = vel_r_previous
    mpos_h = pos_h_previous
    mpos_r = pos_r_previous

    vel_h = vel_h_previous
    vel_r = vel_r_previous
    pos_h = pos_h_previous
    pos_r = pos_r_previous

    linear_velocity_robot = []
    linear_velocity_human = []

    lin_vel_sim_robot = []
    lin_vel_sim_human = []

    total_resample = 0

    for i in range(3000):

        Jh = utils.calculate_jacobian(human_id, pos_h_previous) 
        Jr = utils.calculate_jacobian(robot_id, pos_r_previous) 

        # --- calculate mass matrix for human and robot at current time step--- #
        Mh = utils.calculate_mass_matrix(human_id, pos_h_previous)
        Mr = utils.calculate_mass_matrix(robot_id, pos_r_previous)

        # if using simplified N term then calculate N vector for human and robot at current time step
        Nr = utils.calculate_N_vector(robot_id, pos_r_previous, vel_r_previous)
        Nh = utils.calculate_N_vector(human_id, pos_h_previous, vel_h_previous)

        # ---- check that jacobians are well-defined --- #
        if not (np.allclose(np.linalg.pinv(Jh) @ Jh, np.eye(len(utils.get_good_joints(human_id)))) and \
            np.allclose(np.linalg.pinv(Jr) @ Jr, np.eye(len(utils.get_good_joints(robot_id)))) and \
                np.allclose(np.linalg.pinv(Mh) @ Mh, np.eye(len(utils.get_good_joints(human_id))))):
            raise ValueError(f"Bad jacobian, time step {i}")
        
        human_lin_vel = utils.get_linear_velocity(Jh, vel_h_previous)
        robot_lin_vel = utils.get_linear_velocity(Jr, vel_r_previous)

        linear_velocity_robot.append(robot_lin_vel)
        linear_velocity_human.append(human_lin_vel)

        _,_,_,_,_,_, human_sim_lin_vel,_ = p.getLinkState(human_id, p.getNumJoints(human_id) - 1, computeLinkVelocity=True)
        _,_,_,_,_,_, robot_sim_lin_vel,_  = p.getLinkState(robot_id, p.getNumJoints(robot_id) - 1, computeLinkVelocity=True)

        lin_vel_sim_robot.append(robot_sim_lin_vel)
        lin_vel_sim_human.append(human_sim_lin_vel)

        while True:

            if control == "robot_torque":
                tau_robot = (np.random.sample((len(utils.get_good_joints(robot_id)),1)) - np.ones((len(utils.get_good_joints(robot_id)),1))*0.5) * 10.0
                tau_robot_flat = np.squeeze(tau_robot)

                
                big_R = np.eye(6) # rotation to apply to twist from robot linear velocities (J_r @ qdot_r)
                big_R[:3, :3] = robot_to_human_ee
                big_R[3:, 3:] = robot_to_human_ee

                acc_r = utils.get_u_N(Jr, Jh, Mh, np.expand_dims(Nh, axis=1), Mr, np.expand_dims(Nr, axis=1), delta_t, tau_robot, np.expand_dims(vel_h_previous, axis=1), np.expand_dims(vel_r_previous, axis=1), big_R)
                np.zeros(acc_r.shape)
                acc_r = np.squeeze(acc_r)
                vel_r = vel_r_previous + acc_r * delta_t
                pos_r = pos_r_previous + vel_r * delta_t
                robot_lin_vel = Jr @ vel_r 

                human_lin_vel = big_R @ robot_lin_vel
                vel_h = np.linalg.pinv(Jh) @ human_lin_vel
                pos_h = pos_h_previous + vel_h * delta_t
                acc_h = (vel_h - vel_h_previous) / delta_t 

                
                new_out = math_dynamics.step(tau_robot_flat)

                acc_r = new_out.active_qdd
                vel_r = new_out.active_qd
                pos_r = new_out.active_q
                
                acc_h = new_out.passive_qdd
                vel_h = new_out.passive_qd
                pos_h = new_out.passive_q

                break
            elif control == "robot_acceleration":
                acc_r = (np.random.sample((len(utils.get_good_joints(robot_id)),)) - np.ones((len(utils.get_good_joints(robot_id)),))*0.5) * 5.0
                vel_r = vel_r_previous + acc_r * delta_t
                pos_r = pos_r_previous + vel_r * delta_t

                robot_lin_vel = Jr @ vel_r
                human_lin_vel = robot_lin_vel
                vel_h = np.linalg.pinv(Jh) @ human_lin_vel
                pos_h = pos_h_previous + vel_h * delta_t
                acc_h = (vel_h - vel_h_previous) / delta_t

                tau_human = Mh @ acc_h + Nh
                eef_robot = np.linalg.pinv(Jh.T) @ tau_human
                tau_robot = Mr @ acc_r + Nr + Jr.T @ eef_robot
                
                # if any element of tau_robot is greater than 50 then continue
                if np.all(np.abs(tau_robot) < 100):
                    break
                else:
                    print("tau_robot: ", tau_robot)
                    print("tau_robot too high, trying again for i: ", i)
                    total_resample += 1

            elif control == "ee_force":
                eef_robot = (np.random.sample((6,)) - np.ones((6,)))*0.05
                eef_robot[3:] = eef_robot[3:] * 0.01

                acc_h = np.linalg.pinv(Mh) @ (Jh.T @ eef_robot - Nh)
                vel_h = vel_h_previous + acc_h * delta_t
                pos_h = pos_h_previous + vel_h * delta_t

                human_lin_vel = Jh @ vel_h
                robot_lin_vel = human_lin_vel
                vel_r = np.linalg.pinv(Jr) @ robot_lin_vel
                pos_r = pos_r_previous + vel_r * delta_t
                acc_r = (vel_r - vel_r_previous) / delta_t

                tau_robot = Mr @ acc_r + Nr + Jr.T @ eef_robot

                break
                # if any element of tau_robot is greater than 50 then continue
                if np.all(np.abs(tau_robot) < 10):
                    break
                else:
                    print("tau_robot: ", tau_robot)
                    print("tau_robot too high, trying again for i: ", i)
                    total_resample += 1

        #set current velocities and positions to previous for next time step
        vel_h_previous = vel_h
        vel_r_previous = vel_r
        pos_r_previous = pos_r
        pos_h_previous = pos_h

        # --- in order to use torque control, velocity control must be disabled at every time step --- #
        # for j in utils.get_good_joints(robot_id):
        #     p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)
        # for j in utils.get_good_joints(human_id):
        #     p.setJointMotorControl2(human_id, j, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControlArray(robot_id, utils.get_good_joints(robot_id), p.TORQUE_CONTROL, forces=tau_robot)
            
        torques.append(tau_robot)

        mvel_h_previous = mvel_h
        mvel_r_previous = mvel_r
        mpos_h_previous = mpos_h
        mpos_r_previous = mpos_r
        
        p.stepSimulation()
        # time.sleep(0.01)

        # append predicted joint states
        velocity_h.append(vel_h)
        velocity_r.append(vel_r)
        position_h.append(pos_h)
        position_r.append(pos_r)
        acc_r_sim.append(acc_r)

        # get actual joint states for robot and human in simulation and save to list
        mpos_r, mvel_r, mtorq_r, mforce_r = utils.getMotorJointStates(robot_id)
        mvel_r = np.array(mvel_r)
        mpos_r = np.array(mpos_r)
        velocity_r_sim.append(mvel_r)
        position_r_sim.append(mpos_r)
        
        mpos_h, mvel_h, mtorq, mforce_h = utils.getMotorJointStates(human_id)
        mvel_h = np.array(mvel_h)
        mpos_h = np.array(mpos_h)
        velocity_h_sim.append(mvel_h)
        position_h_sim.append(mpos_h)

        macc_r = (mvel_r - mvel_r_previous) / delta_t
        macc_h = (mvel_h - mvel_h_previous) / delta_t

        
    print("Total resample: ", total_resample)
    print("End test")
    p.stopStateLogging(log_id)

    plotLinVelSim(np.array(lin_vel_sim_human), np.array(lin_vel_sim_robot))

    return np.array(acc_r_sim), np.array(torques), np.array(velocity_h), np.array(velocity_r), np.array(velocity_h_sim), np.array(velocity_r_sim), np.array(position_h), np.array(position_r), np.array(position_h_sim), np.array(position_r_sim), np.array(linear_velocity_human), np.array(linear_velocity_robot)

def main():

    args = parser.parse_args()
    global DEBUG
    DEBUG = args.debug

    #load simulation
    # sim_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,0.0)
    ground_plane_id = p.loadURDF("plane.urdf")
    p.setRealTimeSimulation(0)

    p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG,
                            globalCFM=0.000001)

    # --- set robot and human arm starting positions --- #
    robot_init_pos = [0.8, -0.1, 0.5]
    robot_init_orn_obj = R.from_euler('xyz', [0,0,np.pi]) # rotate 180 degrees around z axis
    robot_init_orn = robot_init_orn_obj.as_quat()
    # robot_init_config = [1.00, -1.0, -2.0, 0.00, 1.57, 0.78]
    
    human_init_pos = [0.15, 0.1, 1.4]
    human_init_orn_obj = R.from_euler('xyz', [np.pi,0,0])
    human_init_orn = human_init_orn_obj.as_quat()
    # human_init_config = [1.00, -1.0, -2.0, 0.5, 0.0, 0.0, 0,0]


    # robot base to human base transform
    transform_mat = human_init_orn_obj.as_matrix().T @ robot_init_orn_obj.as_matrix()
    transform_obj = R.from_matrix(transform_mat)


    # --- load robot and human arm --- #
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # load robot arm
    franka_urdf_path = os.path.join(cur_dir, '../storm/content/assets/urdf/franka6/panda.urdf')
    human_urdf_path = os.path.join(cur_dir, '../storm/content/assets/urdf/human_arm/arm_6dof_continuous.urdf')
    
    robot_id = p.loadURDF(franka_urdf_path, robot_init_pos, robot_init_orn, useFixedBase=True, flags = p.URDF_USE_INERTIA_FROM_FILE)
    
    # load human but replace with robot urdf
    human_id = p.loadURDF(human_urdf_path, human_init_pos, human_init_orn, useFixedBase=True, flags = p.URDF_USE_INERTIA_FROM_FILE)
    

    # set human arm joint positions
    human_initial_positions = [-2.89089697, -0.9754231 , -2.08760543, -1.08786023,  0.14448669, -0.26559232]
    for i, joint_id in enumerate(utils.get_good_joints(human_id)):
        p.resetJointState(human_id, joint_id, human_initial_positions[i])

    robot_initial_positions = [-0.32841343,  0.28639087, -1.99012555, -1.88232461,  1.32335729, -0.00612142]
    for i, joint_id in enumerate(utils.get_good_joints(robot_id)):
        p.resetJointState(robot_id, joint_id, robot_initial_positions[i])


    # --- set some constants --- #
    panda_n_joints = p.getNumJoints(robot_id)
    human_n_joints = p.getNumJoints(human_id)

    robot_ee_link = panda_n_joints - 1
    human_ee_link = human_n_joints - 1


    # --- create fixed joint constraint between panda and human arm at end effector of both --- #
    ###### THE rotation matrix for human ee to robot ee OR robot ee to human ee (cuz inverse (transpose) is the same) ######
    grasp_orn_mat = np.array([[0, 0, -1], 
                              [0, -1, 0], 
                              [-1, 0, 0]])
    grasp_orn_obj = R.from_matrix(grasp_orn_mat)

    cid = p.createConstraint(
        robot_id, robot_ee_link, human_id, human_ee_link, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0], [0,0,0,1], grasp_orn_obj.as_quat())    
    
    # --- remove friction terms as well contact stiffness and damping --- #
    for i in range(p.getNumJoints(human_id)):
        p.changeDynamics(human_id, i, jointDamping=0.0, anisotropicFriction=0.0,maxJointVelocity=5000, linearDamping=0.0, angularDamping=0.0, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0, contactStiffness=0.0, contactDamping=0.0) #, jointLowerLimit=-6.283185 * 500, jointUpperLimit=6.283185 * 500)
    
    for i in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, i, jointDamping=0.0, anisotropicFriction=0.0, maxJointVelocity=5000, linearDamping=0.0, angularDamping=0.0, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0, contactStiffness=0.0, contactDamping=0.0), #, jointLowerLimit=-6.283185 * 200, jointUpperLimit=6.283185 * 200)


    # --- remove collision for both robot and human arms --- #
    group = 0
    mask=0 
    for linkIndex in range(p.getNumJoints(human_id)):
        p.setCollisionFilterGroupMask(human_id, linkIndex, group, mask)
    for linkIndex in range(p.getNumJoints(robot_id)):
        p.setCollisionFilterGroupMask(robot_id, linkIndex, group, mask)

    # --- step simulation to enforce constraint (does not update automatically) ---#
    for i in range(200):
        p.stepSimulation()
    # time.sleep(0.5)

    # --- apply velocity control to panda arm to make it stationary --- #
    for i in range(panda_n_joints):
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=50)
    for i in range(human_n_joints):
        p.setJointMotorControl2(human_id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=50)
    
    for i in range(1000):
        p.stepSimulation()

    # --- try to force panda arm to stop moving by setting forces to zero to k timesteps --- #
    for i in range(panda_n_joints):
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(robot_id, i, p.TORQUE_CONTROL, force=0)
    for i in range(human_n_joints):
        p.setJointMotorControl2(human_id, i, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(human_id, i, p.TORQUE_CONTROL,force=0)
    
    for i in range(1000):
        p.stepSimulation()

    robot_ee_position, robot_ee_orientation = utils.getEndEffectorState(robot_id)
    human_ee_position, human_ee_orientation = utils.getEndEffectorState(human_id)

    # transform_robot_ee_to_human = utils.getTransform(robot_ee_position, robot_ee_orientation, human_ee_position, human_ee_orientation)
    # if DEBUG: 
    #     print("transform_robot_ee_to_human after setting constraint: ", transform_robot_ee_to_human)

    # # --- run test control output in current simulation environment --- #
    acc_r_sim, torques, velocity_h, velocity_r, velocity_h_sim, \
        velocity_r_sim, position_h, position_r, position_h_sim, \
            position_r_sim, lin_vel_human, lin_vel_robot = test_control_output(cid, human_id, robot_id, delta_t=args.delta, control=args.control, robot_to_human_ee=transform_mat)
    

    # --- save results --- #
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # load robot arm
    figure_dir = os.path.join(cur_dir, "../figures/dynamics/")

    save_vel_only_figs = False

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    # --- plot and save results --- #
    for i in range(p.computeDofCount(robot_id)):
        figure, axis = plt.subplots(2, 1, figsize=(10,7))
        axis[1].plot(acc_r_sim[:, i], label=f"U Value joint {i + 1}", color="red")
        axis[0].plot(torques[:, i], label=f"Torque joint {i + 1}")
        axis[0].set_ylabel("Torque (Nm)")
        axis[0].set_title(f"Torque for joint {i + 1}")
        axis[1].set_ylabel("Acceleration - u (rad/s^s) ")
        axis[1].set_title(f"Acceleration (u) for joint {i + 1}")
        axis[1].set_xlabel("Time")
        plt.savefig(os.path.join(figure_dir, f"robot_torque_acc_joint_{i + 1}.png"))
        plt.close()

        figure, axis= plt.subplots(2, 1, figsize=(10,7))
        axis[1].plot(velocity_r[:, i], label=f"Velocity robot joint {i + 1} calculated")
        axis[1].plot(velocity_r_sim[:, i], label=f"Velocity robot joint {i + 1} measured sim")
        axis[0].plot(position_r[:, i], label=f"Position robot joint {i + 1} calculated")
        axis[0].plot(position_r_sim[:, i], label=f"Position robot joint {i + 1} measured sim")
        axis[0].set_ylabel("Position (rad)")
        axis[0].set_title(f"Position for joint {i + 1}")
        axis[1].set_ylabel("Velocity (rad/s)")
        axis[1].set_title(f"Velocity for joint {i + 1}")
        axis[1].set_xlabel("Time")
        axis[0].legend()
        axis[1].legend()
        plt.savefig(os.path.join(figure_dir, f"robot_velocity_pos_joint_{i + 1}.png"))
        plt.close()
        if save_vel_only_figs:
            #plt.legend()
            plt.figure(figsize=(9, 6))
            plt.plot(velocity_r[:, i], label=f"Velocity robot joint {i + 1} calculated")
            plt.plot(velocity_r_sim[:, i], label=f"Velocity robot joint {i + 1} measured sim")
            plt.ylabel("Velocity (rad/s)")
            plt.xlabel("Time")
            plt.title(f"Robot Velocities for joint {i + 1}")
            plt.legend()
            #plt.show()
            plt.savefig(os.path.join(figure_dir, f"robot_velocity_joint_{i + 1}.png"))
            plt.close()

    for i in range(p.computeDofCount(human_id)):
        if save_vel_only_figs:
            plt.figure(figsize=(9, 6))
            plt.plot(velocity_h[:, i], label=f"Velocity human joint {i + 1} calculated")
            plt.plot(velocity_h_sim[:, i], label=f"Velocity human joint {i + 1} measured sim")
            plt.ylabel("Velocity (rad/s)")
            plt.xlabel("Time")
            plt.title(f"Human Velocities for joint {i + 1}")
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f"human_velocity_joint_{i + 1}.png"))
            plt.close()

        figure, axis= plt.subplots(2, 1, figsize=(10,7))
        axis[1].plot(velocity_h[:, i], label=f"Velocity human joint {i + 1} calculated")
        axis[1].plot(velocity_h_sim[:, i], label=f"Velocity human joint {i + 1} measured sim")
        axis[0].plot(position_h[:, i], label=f"Position human joint {i + 1} calculated")
        axis[0].plot(position_h_sim[:, i], label=f"Position human joint {i + 1} measured sim")
        axis[0].set_ylabel("Position (rad)")
        axis[0].set_title(f"Position for joint {i + 1}")
        axis[1].set_ylabel("Velocity (rad/s)")
        axis[1].set_title(f"Velocity for joint {i + 1}")
        axis[1].set_xlabel("Time")
        axis[0].legend()
        axis[1].legend()
        plt.savefig(os.path.join(figure_dir, f"human_velocity_pos_joint_{i + 1}.png"))


    vel_h_x = lin_vel_human[:, 0]
    vel_h_y = lin_vel_human[:, 1]
    vel_h_z = lin_vel_human[:, 2]

    vel_r_x = lin_vel_robot[:, 0]
    vel_r_y = lin_vel_robot[:, 1]
    vel_r_z = lin_vel_robot[:, 2]

    #plot linear velocities
    fig, ax = plt.subplots(3, 1, figsize=(12,10))
    ax[0].plot(vel_h_x, label="Human x velocity")
    ax[0].plot(vel_r_x, label="Robot x velocity")
    ax[0].set_ylabel("Velocity (m/s)")
    ax[0].set_title("X Velocity")
    ax[0].legend()
    ax[1].plot(vel_h_y, label="Human y velocity")
    ax[1].plot(vel_r_y, label="Robot y velocity")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].set_title("Y Velocity")
    ax[1].legend()
    ax[2].plot(vel_h_z, label="Human z velocity")
    ax[2].plot(vel_r_z, label="Robot z velocity")
    ax[2].set_ylabel("Velocity (m/s)")
    ax[2].set_title("Z Velocity")
    ax[2].legend()

    plt.savefig(os.path.join(figure_dir, "linear_velocities.png"))


    input('Press enter to exit')
    # while True:
    #     p.stepSimulation() # --- step simulation --- #


def plotLinVelSim(lin_vel_human, lin_vel_robot):

    vel_h_x = lin_vel_human[:, 0]
    vel_h_y = lin_vel_human[:, 1]
    vel_h_z = lin_vel_human[:, 2]

    vel_r_x = lin_vel_robot[:, 0]
    vel_r_y = lin_vel_robot[:, 1]
    vel_r_z = lin_vel_robot[:, 2]

    #plot linear velocities
    fig, ax = plt.subplots(3, 1, figsize=(12,10))
    ax[0].plot(vel_h_x, label="Human x velocity")
    ax[0].plot(vel_r_x, label="Robot x velocity")
    ax[0].set_ylabel("Velocity (m/s)")
    ax[0].set_title("X Velocity")
    ax[0].legend()
    ax[1].plot(vel_h_y, label="Human y velocity")
    ax[1].plot(vel_r_y, label="Robot y velocity")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].set_title("Y Velocity")
    ax[1].legend()
    ax[2].plot(vel_h_z, label="Human z velocity")
    ax[2].plot(vel_r_z, label="Robot z velocity")
    ax[2].set_ylabel("Velocity (m/s)")
    ax[2].set_title("Z Velocity")
    ax[2].legend()

    # plt.show()

if __name__ == '__main__':
    main()
       
