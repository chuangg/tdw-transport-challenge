from typing import List

from magnebot import Arm
from magnebot import ActionStatus as TaskStatus
from transport_challenge import Transport

from tdw.tdw_utils import TDWUtils

import numpy as np
import math
import random
from json import loads

from pkg_resources import resource_filename
import pickle

find_reward = 3
grasp_container_reward = 5
grasp_object_reward = 10
drop_object_reward = 15

class Basic_controller(Transport):

    def __init__(self, port: int = 1071, demo=False, \
                        screen_size = 128, physics=True, \
                        train = 0, exp = False, fov = 90, launch_build=False):
        """
        :param port: The port number.
        """

        super().__init__(port=port, launch_build=launch_build, \
                screen_width=screen_size, screen_height=screen_size, fov = fov)
        self.demo = demo
        self.physics = physics
        self.exp = exp
        self.port = port
        self.reward = 0
        self.step_reward = 0
        
        
        self.finish = {}
        self.held = {}
        self.known_object_ids = []
        self.goal_id = []
        
        self.content_container = {}
        
        self._held_objects = {Arm.left: [], Arm.right: []}
        self.teleport_num = 0
        
        self.find_goal = 0
        self.find_object = 0
        self.find_container = 0
        
        self.pre_known = 0
        
        self.total_target_object = 0
        
        self.FOV = fov
        path = resource_filename(__name__, '')
        
        self.sensor_noise_fwd = \
                pickle.load(open(f"{path}/sensor_noise_fwd.pkl", 'rb'))
        
        
    def noop(self):
        return TaskStatus.success
            
    def _move_forward(self, distance=0.5):
        '''
        move forward at 0.5
        '''        
        return self.move_by(0.5, arrived_at = 0.1)        
                                    
    def _turn(self, left=True, angle=15):
        '''
        turn left or right at 15
        '''
        
        if left:
            angle = -angle
        else:
            angle = angle
        
        return self.turn_by(angle=angle,
                            aligned_at=2)
    
    def _go_to(self, object):
        '''
        go to an object based on an object id
        '''
        status = self.turn_to(int(object))
        object = int(object)
        p1 = self.state.object_transforms[object].position
        p2 = self.state.magnebot_transform.position
        d = ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
        
        if  d > 3:            
            return TaskStatus.failed_to_move
        status = self.move_to(int(object), arrived_at=0.33)
        return status
    
    def _reach_for_target(self, arm, target):
        pass        
            
    
    def teleport_object(self, object_id):
        '''
        do not use this function
        '''
        self.teleport_num += 0.3
        position = {'x': 20 + self.teleport_num, 'y': 3, 'z': 20 + self.teleport_num}
        
        self.communicate([{"$type": "teleport_object",
               "position": position,
               "id": int(object_id),
               "physics": True}])
    
    def _drop(self):
        '''
        drop held objects by two arms
        '''
        held_o = []
        if self.physics:
            held_o.extend(self.state.held[Arm.left])
            held_o.extend(self.state.held[Arm.right])
        else:
            flag_drop = False
            agent_position = self.state.magnebot_transform.position
            st = (agent_position[0], agent_position[2])
            for id in self.goal_object_ids:
                object_position = self.state.object_transforms[id].position
                gp = (object_position[0], object_position[2])
                if self.l2_distance(st, gp) < 3:
                    flag_drop = True
            if not flag_drop:
                return TaskStatus.not_holding
            held_o = self._held_objects[Arm.left] + \
                        self._held_objects[Arm.right]

        self.held_container = None
        for object_id in held_o:
            self.update_finish(object_id)
            
            if object_id in self.containers:
                for o in self.content_container[object_id]:
                    self.update_finish(o)
        if self.physics:
            for arm in [Arm.left, Arm.right]:
                for o in self.state.held[arm]:
                    self.drop(int(o), arm)
        else:
            self._held_objects[Arm.left] = []
            self._held_objects[Arm.right] = []
        return TaskStatus.success
            
    def check_grasp(self, object_id):
        x, y, z = self.state.magnebot_transform.position
        ox, oy, oz = self.state.object_transforms[object_id].position
        return ((x - ox) ** 2 + (z - oz) ** 2) ** 0.5 < 0.7
    
    def _grasp(self, object_id = None, arm = None):
        '''
        grasp an object by an arm
        '''
        if not self.physics:
            if object_id is None:
                tot_objects = []
                if self.held_container is None:
                    tot_objects.extend(self.containers)
                tot_objects.extend(self.target_objects)

                for o in tot_objects:
                    if self.check_grasp(o):
                        object_id = o
                        break
            
            if object_id is None or not self.check_grasp(object_id):
                return TaskStatus.failed_to_grasp
            self.teleport_object(object_id)
            if len(self._held_objects[Arm.left]) == 0:
                arm = Arm.left
            elif len(self._held_objects[Arm.right]) == 0:
                arm = Arm.right
            else:
                return TaskStatus.failed_to_grasp
            self._held_objects[arm].append(object_id)
            
            self.update_grasp(object_id)
            return TaskStatus.success
        else:
            object_id = int(object_id)
            status = self.pick_up(target=object_id,
                                    arm=arm)
            if object_id in self.state.held[arm]:
                
                status = TaskStatus.success                
            else:
                status = TaskStatus.failed_to_grasp
            if status == TaskStatus.success:
                self.update_grasp(object_id)
            
            return status
    
    def _put_in_container(self, object_id = None, container_id = None):
        '''
        put object_id into container_id
        '''
        if not self.physics:
            if self.held_container is None:
                return TaskStatus.not_in
            for arm in [Arm.left, Arm.right]:
                for o in self._held_objects[arm]:
                    if o in self.target_objects:
                        object_id = o
            if object_id is None:
                return TaskStatus.not_in
            self.content_container[self.held_container].append(object_id)
            return TaskStatus.success
        else:
            if len(self.content_container[container_id]) > 2:
                return TaskStatus.not_in
            status = self.put_in()
            if status == TaskStatus.success:
                self.content_container[container_id].append(object_id)
            
            return status
    
    def get_object_pos(self, id):
        x, y, z = self.state.object_transforms[id].position
        _, i, j, _ = self.check_occupied(x, z)
        return i, j
    
    
    def update_grasp(self, object_id):
        if object_id in self.containers:            
            if object_id not in self.grasped:
                self.step_reward += grasp_container_reward
            self.held_container = object_id
            if object_id not in self.content_container:
                self.content_container[object_id] = []
        if object_id in self.target_objects:
            if object_id not in self.grasped:
                self.step_reward += grasp_object_reward
            self.num['grasp'] += 1
        self.grasped[object_id] = 1
    
    def cal_dis(self, id1, id2):
        p1 = self.state.object_transforms[id1].position
        p2 = self.state.object_transforms[id2].position
        d = ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
        return d
    
    def check_goal_zone(self, object_id):
        if self.physics:
            for id in self.goal_object_ids:
                if self.cal_dis(object_id, id) < 3:
                    return True
            return False
        else:
            return True
    
    def update_finish(self, object_id):
        if object_id in self.containers:
            pass
        if object_id in self.target_objects \
                and self.check_goal_zone(object_id) \
                and object_id not in self.finish:
            self.step_reward += drop_object_reward
            self.num['finish'] += 1
            self.finish[object_id] = 1
        
    def _obs(self):   
        obs = {}
        obs['rgb'] = np.array(self.state.get_pil_images()['img']).transpose(2, 0, 1)    #3, 128, 128
        
        obs['seg_mask'] = np.array(self.state.get_pil_images()['id'])
        
        obs['depth'] = TDWUtils.get_depth_values(self.state.images['depth'], \
                        width = 128, 
                        height = 128)   #128 * 128
                        
        x, y, z = self.state.magnebot_transform.position
        noise_x, noise_z, noise_o = self.sensor_noise_fwd.sample()[0][0]
        x, y, z = x + noise_x, y, z + noise_z

        fx, fy, fz = self.state.magnebot_transform.forward
        obs['agent'] = [x, y, z, fx, fy, fz]
        
        obs['held_objects'] = []
        if self.physics:
            for arm in [Arm.left, Arm.right]:
                if len(self.state.held[arm]) == 0:
                    obs['held_objects'].append(None)
                else:
                    obs['held_objects'].append(self.object_ids.index(self.state.held[arm][0]))

        visible_objects = self.get_visible_objects()
        obs['visible_objects'] = [{
            'object_id': self.object_ids.index(o),
            'type': self.get_object_type(o),
            'seg_color': self.objects_static[o].segmentation_color}
                        for o in visible_objects]
        for _ in range(len(visible_objects), 20):
            obs['visible_objects'].append({'object_id': None, 'type': None, 'seg_color': None})  
        obs['FOV'] = self.FOV
        obs['camera_matrix'] = np.array(self.state.camera_matrix).reshape((4, 4))
        
        
        return obs
    
    def _info(self, env):
        '''
        return the env's info
        '''
        info = {}
        
        info['map_size'] = (120, 60)
        info['_scene_bounds'] = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        x, y, z = self.state.magnebot_transform.position
        noise_x, noise_z, noise_o = self.sensor_noise_fwd.sample()[0][0]
        x, y, z = x + noise_x, y, z + noise_z
        info['position'] = [x, y, z]
        info['objects_info'] = self.objects_static
        info['target_objects'] = self.target_objects
        info['containers'] = self.containers
        info['camera_info'] = {"FOV": self.FOV, \
                                "camera_matrix": self.state.camera_matrix}
        info['depth'] = TDWUtils.get_depth_values(self.state.images['depth'], \
                        width = 128, 
                        height = 128)   #128 * 128
        info['pc'] = TDWUtils.get_point_cloud(depth = info['depth'], \
                        vfov = self.FOV, \
                        camera_matrix = info["camera_info"]["camera_matrix"])
        
        '''info['content_container'] = self.content_container
        info['container_full'] = False
        if self.held_container is not None:
            info['container_full'] = len(self.content_container[self.held_container]) > 2'''
        visible_objects = self.get_visible_objects()
        info['visible_objects'] = [{
            'id': o,
            'object_id': self.object_ids.index(o),
            'type': self.get_object_type(o),
            'position': self.state.object_transforms[o].position,
            'seg_color': self.objects_static[o].segmentation_color}
                        for o in visible_objects]   #20
        
        info['finish'] = self.num["finish"]
        
        info['port'] = self.port
        if self.exp:            
            for o in visible_objects:
                if o not in self.look:
                    self.look[o] = 1
                    if o in self.containers or \
                    o in self.target_objects or \
                    o in self.goal_object_ids:
                        self.step_reward += find_reward
        #only for debug and need remove
        #info['object_transforms'] = self.state.object_transforms
        return info    
    
    def get_object_type(self, id):        
        if id in self.target_objects:
            return 0
        if id in self.containers:
            return 1
        if self.objects_static[id].category == 'bed':
            return 2
        return 3
    
    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5
    
    def check_goal(self, thresold = 1.5):        
        reward = self.step_reward
        
        return reward
        
    def _end_task_status(self, task_status, f):
        #self._previous_collision = None
        if task_status == TaskStatus.collision:
            for colliding_object in self.colliding_objects:
                f.write(f'collision: {self.objects_static[colliding_object].category},{self.objects_static[colliding_object].mass}\n')
                f.flush()

                if colliding_object in self.containers or colliding_object in self.target_objects:
                    self._previous_collision = None
