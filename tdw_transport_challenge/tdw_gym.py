import gym
from gym.core import Env
import numpy as np
import os
import time
from tdw_transport_challenge.controller import Basic_controller as Controller
from tdw_transport_challenge.utils import create_tdw, kill_tdw

from magnebot import Arm
from magnebot import ActionStatus as TaskStatus
from json import loads

import random
import pickle
from pkg_resources import resource_filename
import shlex
import subprocess

class TDW(Env):
    
    def __init__(self, port = 1071, ip_address=None, demo=False, physics = True, \
                        rank=0, num_scenes = 0, train = 0, \
                        screen_size = 128, exp = False, launch_build_inside_docker=False):
        self.physics = physics
        self.exp = exp
        self.rank = rank
        self.num_scenes = num_scenes
        self.data_id = rank     
        self.train = train
        self.port = port
        self.docker_id = None
        if launch_build_inside_docker:
            p = subprocess.Popen(shlex.split(f'./TDW/TDW.x86_64 -port={port}'))
        else:
            if ip_address is not None:
                self.docker_id = create_tdw(port=port)
                print('connect:', self.docker_id, port)
            else:
                print(f'connect: {port}')

        self.screen_size = screen_size
        self.controller = Controller(port=port, demo=demo, \
                            screen_size = self.screen_size, physics = physics, \
                            train = train, exp = exp, fov = 90)
        print("Controller connected")
        rgb_space = gym.spaces.Box(0, 256,
                                 (3,
                                  self.screen_size,
                                  self.screen_size), dtype=np.int32)
        seg_space = gym.spaces.Box(0, 256, \
                                (self.screen_size, \
                                self.screen_size, \
                                3), dtype=np.int32)
        depth_space = gym.spaces.Box(0, 256, \
                                (self.screen_size, \
                                self.screen_size), dtype=np.int32)
        object_space = gym.spaces.Dict({
            'object_id': gym.spaces.Discrete(30),
            'type': gym.spaces.Discrete(4),
            'seg_color': gym.spaces.Box(0, 255, (3, ), dtype=np.int32)
        })
        self.observation_space = gym.spaces.Dict({
            'rgb': rgb_space,
            'seg_mask': seg_space,
            'depth': depth_space,
            'agent': gym.spaces.Box(-30, 30, (6, ), dtype=np.float32),
            'held_objects': gym.spaces.Tuple((gym.spaces.Discrete(30), gym.spaces.Discrete(30))),
            'visible_objects': gym.spaces.Tuple(object_space for _ in range(20)),
            'status': gym.spaces.Discrete(2),
            'FOV': gym.spaces.Box(0, 120, (1,), dtype=np.float32),
            'camera_matrix': gym.spaces.Box(-30, 30, (4, 4), dtype=np.float32)
        })
        
        self.action_space = gym.spaces.Dict({
            'type': gym.spaces.Discrete(6),
            'object': gym.spaces.Discrete(30),
            'arm': gym.spaces.Discrete(2)
        })
        if self.exp:
            self.max_step = 400
        else:
            self.max_step = 1000
        self.f = open(f'action{port}.log', 'w')
        self.action_list = []
        if self.train == 0:
            path = resource_filename(__name__, "test_dataset.pkl")
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
            random.shuffle(self.data)
            self.data_id = 0
            self.data_n = len(self.data)
            
        #debug
        path = f'./debug_images{self.port}'
        import shutil

        if os.path.isdir(path):
            shutil.rmtree(path)
        
    def reset(self, scene_info = None):        
        '''
        reset the environment
        input:
            data_id: reset based on the data_id
        '''
        start = time.time()
        scene = "5a"
        layout = 0
        seed = None
        if self.train == 0 and scene_info is None:
            scene_info = self.data[self.data_id % self.data_n]
            self.data_id += 1
        if scene_info is not None:
            scene = scene_info['scene']
            layout = scene_info['layout']
            seed = scene_info['seed']
        self.scene_info = scene_info
        self.controller.init_scene(scene=scene, layout=layout, room = 5, random_seed=seed)
        self.controller.object_ids = []
        for id in self.controller.objects_static:
            self.controller.object_ids.append(id)            
        
        self.controller.reward = 0
        self.num_step = 0
        
        self.controller.teleport_num = 0
        
        self.controller.num = {
            'grasp': 0,
            'finish': 0
        }
        
        self.controller.grasped = {}
        self.controller.finish = {}
        self.controller.look = {}
        
        self.controller.content_container = {}
        
        self.controller._held_objects = {Arm.left: [], Arm.right: []}
        self.controller.held_container = None
        
        self.goal_object = "bed"
        self.controller.goal_object_ids = []                
        flag = False        
        
        for id in self.controller.objects_static:
            o = self.controller.objects_static[id]
            if o.category == self.goal_object:
                self.controller.goal_object_ids.append(id)
                flag = True                

        info = self.controller._info(self)
        info['num_step'] = 0
        info['goal_object'] = "bed"
        self.done = False
        self.num_wrong_status = 0

        
        return self.controller._obs(), info
        
    def step(self, action):
        '''
        Run one timestep of the environment's dynamics
        '''
        start = time.time()
        self.controller.step_reward = 0
        
        if not isinstance(action, dict):
            a = {}
            a["type"] = action
            action = a
            
        if "object" in action:
            action["object"] = self.controller.object_ids[action["object"]]
        if "container" in action:
            action["container"] = self.controller.object_ids[action["container"]]
        task_status = TaskStatus.success
        if action["type"] == 0:       #move forward
            task_status = self.controller._move_forward()
        elif action["type"] == 1:     #turn left
            task_status = self.controller._turn(left=True, angle = 15)
        elif action["type"] == 2:     #turn right
            task_status = self.controller._turn(left=False, angle = 15)
        elif action["type"] == 3:       #go to and grasp            
            if self.physics:
                task_status = self.controller._go_to(action["object"])
            
            if action['arm'] == 'left':
                action['arm'] = Arm.left
            elif action['arm'] == 'right':
                action['arm'] = Arm.right
            task_status = self.controller._grasp(action["object"], \
                                                action["arm"])        
        elif action["type"] == 4:       #put in container
            task_status = self.controller._put_in_container(
                                    action["object"], \
                                    action["container"])
        elif action["type"] == 5:      #drop
            task_status = self.controller._drop()  
        else:
            assert False
        self.f.write('step: {}, action: {}, time: {}, status: {}\n'
                .format(self.num_step, action["type"],
                time.time() - start,
                task_status))
        self.f.write('position: {}, forward: {}, container: {}\n'.format(
                self.controller.state.magnebot_transform.position,
                self.controller.state.magnebot_transform.forward,
                self.controller.content_container))
        self.f.flush()
        self.action_list.append(action)
        reward = self.controller.check_goal()
        if task_status != TaskStatus.success:
            reward -= 0.1
        self.num_step += 1        
        self.controller.reward += reward
        self.controller._end_task_status(task_status, self.f)
        done = False
        if self.num_step >= self.max_step:
            done = True
            self.done = True
        
        obs = self.controller._obs()
        info = self.controller._info(self)
        obs['status'] = task_status == TaskStatus.success
        info['status'] = task_status == TaskStatus.success
        info['done'] = done

        info['num_step'] = self.num_step
        if done:
            info['reward'] = self.controller.reward

        return obs, reward, done, info
     
    def render(self):
        return None
        
    def save_images(self, dir='./Images'):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.controller.state.save_images(dir)
    
    def seed(self, seed=None):
        self.seed = np.random.RandomState(seed)
    
    def close(self):
        print('close')
        with open(f'action.pkl', 'wb') as f:
            d = {'scene_info': self.scene_info, \
                'actions': self.action_list}
            pickle.dump(d, f)
        if self.docker_id is not None:
            kill_tdw(self.docker_id)