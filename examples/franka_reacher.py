#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym 

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#



import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import math

import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array, as_euler_angles, quaternion

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)

def rotation_matrix_to_quaternion(R): # Rotation matrix to quaternion
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2  # S=4*qw 
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S 
        qz = (m10 - m01) / S 
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx 
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

def mpc_robot_interactive(args, gym_instance):
    vis_ee_target = True
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'

    
    gym = gym_instance.gym
    sim = gym_instance.sim
    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    if(args.cuda):
        device = 'cuda'
    else:
        device = 'cpu'

    sim_params['collision_model'] = None
    # create robot simulation:
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)

    
    # create gym environment:
    robot_pose = sim_params['robot_pose']
    

    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    device = torch.device('cuda', 0) 

    tensor_args = {'device':device, 'dtype':torch.float32}    

    # spawn camera:
    robot_camera_pose = np.array([1.6,-1.5, 1.8 ,0.707 ,0.0 ,0.0, 0.707])
    q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
    robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])
    
    deg2rad = np.pi/180
    
    a = as_float_array(from_euler_angles(np.pi, 0.0, 0.0))
    
    print("aaaaaaaaaaaaaaaa: ", a)

    
    robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
    
    goal_quat_world = np.array([0.0, 1.0, 0.0, 0.0])
    
    
    # # 월드 기준의 목표 방위 (사원수 -> 회전 행렬)
    # goal_quat_world_tensor = torch.tensor([goal_quat_world[0],  # w
    #                                     goal_quat_world[1],  # x
    #                                     goal_quat_world[2],  # y
    #                                     goal_quat_world[3]]) # z
    # goal_rot_world = quaternion_to_matrix(goal_quat_world_tensor.unsqueeze(0))

    # # 월드에서 로봇으로 변환하는 회전 행렬 부분 (3x3 부분만 추출)
    # w_R_r = quaternion_to_matrix(torch.tensor([w_T_r.r.w, w_T_r.r.x, w_T_r.r.y, w_T_r.r.z]).unsqueeze(0))

    # # 목표 회전을 로봇 기준으로 변환
    # goal_rot_robot = torch.matmul(w_R_r.inverse(), goal_rot_world)

    # # goal_rot_robot이 [1, 3, 3] 형태일 수 있으므로, 이를 3x3 형태로 변환
    # goal_rot_robot = goal_rot_robot.squeeze()  # 차원을 축소하여 [3, 3] 형태로 변환

    # # 변환된 회전 행렬을 다시 사원수로 변환
    # goal_quat_robot = rotation_matrix_to_quaternion(goal_rot_robot.numpy())


    
    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0,3] = w_T_r.p.x
    w_T_robot[1,3] = w_T_r.p.y
    w_T_robot[2,3] = w_T_r.p.z
    w_T_robot[:3,:3] = rot[0]

    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)

    

    
    table_dims = np.ravel([1.5,2.5,0.7])
    cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])
    


    cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    
    table_dims = np.ravel([0.35,0.1,0.8])

    
    
    cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    
    table_dims = np.ravel([0.3,0.1,0.8])
    

    # get camera data:
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs

    
    start_qdd = torch.zeros(n_dof, **tensor_args)

    # update goal:

    exp_params = mpc_control.exp_params
    
    current_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
    ee_list = []
    

    mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

    franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    x_des_list = [franka_bl_state]
    
    ee_error = 10.0
    j = 0
    t_step = 0
    i = 0
    x_des = x_des_list[0]
    
    mpc_control.update_params(goal_state=x_des)

    # spawn object:
    x,y,z = 0.0, 0.0, 0.0
    tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002

    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(x, y, z)
    # object_pose.p = gymapi.Vec3(0.01, 0.01, -0.1)
    object_pose.r = gymapi.Quat(0,0,0, 1)
    
    obj_asset_file = "urdf/mug/movable_mug.urdf" 
    obj_asset_root = get_assets_path()
    
    
    if(vis_ee_target):
        # movable mug에 대한 code
        target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_target_object')
        obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
        obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
        gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        # movable mug에 대한 code

        obj_asset_file = "urdf/mug/mug.urdf"
        obj_asset_root = get_assets_path()

        ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_current_as_mug')
        ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
        tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)


    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    
    # g_q = goal_quat_robot
    
    object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])
    object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
    
    

    object_pose = w_T_r * object_pose
    
    # print("object_pose.p_a : ",object_pose.p, "object_pose.r : ",object_pose.r  )
    # global frame x,y,z --> x,z,y 순임
    object_pose.p = gymapi.Vec3(0.5, 1.6, -0.1)

    # object_pose.r = gymapi.Quat(0.498520, 0.500837, -0.511360, 0.489031)
    # object_pose.r = gymapi.Quat(0.9239, -0.3827,  0.00e+00,  0)
    # q = as_float_array(from_euler_angles(-np.pi/2, np.pi + np.pi/8, -np.pi ))
    # q = as_float_array(from_euler_angles(-np.pi/2, -9*np.pi/10, -np.pi ))
    
    # q = as_float_array(from_euler_angles(-1.59, 1.57, -0.02 ))

    
    # object_pose.r = gymapi.Quat(q[0], q[1], q[2],  q[3])
    # q = np.array([ 0.9999997, 0, 0, 0.0007963 ])
    # q = np.array([ 0.0, 0.0, 0.71, -0.71 ])
    
    # object_pose.r = gymapi.Quat(q[0], q[1],  q[2],  q[3])
    # print("object_pose.r:  ",object_pose.r)
    
    
    
    
    
    # if(vis_ee_target):
    #     gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)
    # n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    # prev_acc = np.zeros(n_dof)
    
    ee_pose = gymapi.Transform()

    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                        rot=w_T_robot[0:3,0:3].unsqueeze(0))     # 3. # calculate current EE pose   # w_T_robot : current robot pose 
    
    # 7.07e-01  7.07e-01 -4.33e-17  4.33e-17
    roll = -np.pi
    pitch = 0
    yaw = np.pi
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Yaw -> Pitch -> Roll 순으로 적용
    R = np.dot(R_z, np.dot(R_y, R_x))
    wrrr = np.array([[-1, 0, -0],[-0, 0, 1],[0, 1, 0]])

    print("w_robot_coord: ",w_robot_coord.translation(), w_robot_coord.rotation())
    A = np.array(w_robot_coord.rotation())
    A = A.reshape(3,3)
    print("AAaa: ",A)
    
    aa = np.dot(A,R)
    
    # print("sssssssssssssssss:", aa)
    roll = 0
    pitch =0
    yaw = 0*np.pi/180
    R_xa = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R_ya = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    R_za = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Yaw -> Pitch -> Roll 순으로 적용
    Ra = np.dot(R_za, np.dot(R_ya, R_xa))
    
    quat = rotation_matrix_to_quaternion(aa) #   1. # Rotation matrix to quaternion
    print("sssssssssssssssssquat:", quat, "\n")
    # print(quat[0], quat[1], quat[2], quat[3])
    quat[1] = 6.12323e-17
    quat[2] = 0
    quat[3] = 0
    quat[0] = 1
    
    # object_pose.r = gymapi.Quat( 0, 0.0, -1.0, -6.12323e-17)
    object_pose.r = gymapi.Quat(-0.183013, -0.683013, 0.683013, -0.183013)   #   2. # represnt Orientation using  gymapi.Quat(quat)
    # object_pose.r = gymapi.Quat(1.0, 0.0, 0, 6.12323e-17)
    # object_pose.r = gymapi.Quat(-0.5, 0.5, 0.5, 0.5)
    print("object_pose.r:  ",object_pose.r)
    
    
    if(vis_ee_target):
        gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    prev_acc = np.zeros(n_dof)

    rollout = mpc_control.controller.rollout_fn
    tensor_args = mpc_tensor_dtype
    sim_dt = mpc_control.exp_params['control_dt']
    
    log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                'qddd_des':[]}

    q_des = None
    qd_des = None
    t_step = gym_instance.get_sim_time()

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    # g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    
    txtbool = False
    first = True
    des_pos = []
    des_quat = []
    current_pos = []
    current_quat = []
    computation_time = []
    

    while(i > -100):
        try:
            gym_instance.step()
            if(vis_ee_target):
                pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))
                pose = copy.deepcopy(w_T_r.inverse() * pose)

                if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                    g_pos[0] = pose.p.x
                    g_pos[1] = pose.p.y
                    g_pos[2] = pose.p.z
                    g_q[1] = pose.r.x
                    g_q[2] = pose.r.y
                    g_q[3] = pose.r.z
                    g_q[0] = pose.r.w

                    mpc_control.update_params(goal_ee_pos=g_pos,
                                              goal_ee_quat=g_q)   # 4. # goal update
            t_step += sim_dt
            
            current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
            
            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)

            filtered_state_mpc = current_robot_state #mpc_control.current_state
            curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity']) #* 0.5
            qdd_des = copy.deepcopy(command['acceleration'])
            
            ee_error = mpc_control.get_current_error(filtered_state_mpc)
             
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            
            # get current pose:
            # 5. # get current pose
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
            
            a2 = ee_pose.r.w
            b2 = ee_pose.r.x
            c2 = ee_pose.r.y
            d2 = ee_pose.r.z
            
            quatt21 = quaternion(b2,c2,d2,a2)
            # print("ee_edit_quat : ", np.array(as_euler_angles(quatt21)))
            
            ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
            
            o_r = copy.deepcopy(w_T_r.inverse()) * copy.deepcopy(object_pose)
            
            a1 = o_r.r.w
            b1 = o_r.r.x
            c1 = o_r.r.y
            d1 = o_r.r.z
            
            quatt1 = quaternion(b1,c1,d1,a1)
            
            # print("obj_edit_quat : ", np.array(as_euler_angles(quatt1)))
            
            a = object_pose.r.w
            b = object_pose.r.x
            c = object_pose.r.y
            d = object_pose.r.z
            
            # quatt = quaternion(a,b,c,d)
            quatt = quaternion(b,c,d,a)

            # print("as_euler_angles: ",as_euler_angles(quatt))
            
            a_p = ee_pose.r.w
            b_p = ee_pose.r.x
            c_p = ee_pose.r.y
            d_p = ee_pose.r.z
            
            quatt_p = quaternion(b_p,c_p,d_p,a_p)
            
            if(vis_ee_target):
                gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))
            
            
            # print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt),
            #       "{:.3f}".format(mpc_control.mpc_dt) )
        
            
            gym_instance.clear_lines()
            top_trajs = mpc_control.top_trajs.cpu().float()#.numpy()
            n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


            top_trajs = w_pts.cpu().numpy()
            color = np.array([0.0, 1.0, 0.0])
            for k in range(top_trajs.shape[0]):
                pts = top_trajs[k,:,:]
                color[0] = float(k) / float(top_trajs.shape[0])
                color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                gym_instance.draw_lines(pts, color=color)
            
            
            
            # 6. # visualization
            robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
            #robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            current_state = command
            
            i += 1
            
            if(t_step  > 20.0):
                if(first == True):
                    txtbool = True
                    first = False
                    
            op1 = object_pose.p.x
            op2 = object_pose.p.y
            op3 = object_pose.p.z
            
            op = np.array([op1, op2, op3])
            
            eep1 = ee_pose.p.x
            eep2 = ee_pose.p.y
            eep3 = ee_pose.p.z

            ep = np.array([eep1, eep2, eep3])

        
            print("obj_quat : ", np.array(as_euler_angles(quatt)))
            print("ee_quat : ", np.array(as_euler_angles(quatt_p)), "\n")
            
            
            print(np.array(as_euler_angles(quatt)))
            print(np.array(as_euler_angles(quatt_p)))
                
            des_pos.append(op)
            des_quat.append(np.array(as_euler_angles(quatt)))
            current_pos.append(ep)
            current_quat.append(np.array(as_euler_angles(quatt_p)))
            computation_time.append(mpc_control.opt_dt)
            if (txtbool == True):
                A = np.array(des_pos)
                B = np.array(des_quat)
                C = np.array(current_pos)
                D = np.array(current_quat)
                E = np.array(computation_time)
                E_re = np.reshape(E,(E.shape[0],-1))
                print("A: ",A.shape, "B: ",B.shape, "C: ",C.shape, "D: ",D.shape, "E: ",E_re.shape)
                arr_ABCD = np.concatenate([A,B, C, D,E_re],axis=1)
                np.savetxt('compare_PM.txt',arr_ABCD,delimiter=',')
                print("save txt!!!!!")
                txtbool = False

            

            
        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
    mpc_control.close()
    return 1 
    
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)
    
    
    mpc_robot_interactive(args, gym_instance)
    
