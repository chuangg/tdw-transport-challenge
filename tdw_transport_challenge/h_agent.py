import os

import numpy as np
import cv2
import pyastar2d as pystar
import random
import time
import math



CELL_SIZE = 0.25
ANGLE = 15

def pos2map(x, z, _scene_bounds):
    i = int(round((x - _scene_bounds["x_min"]) / CELL_SIZE))
    j = int(round((z - _scene_bounds["z_min"]) / CELL_SIZE))
    return i, j



class H_agent:
    def __init__(self, logger):
        self.is_reset = False
        self.logger = logger
        self.map_id = 0
        random.seed(1024)
        self.pre_action = None
        
        self.map_size = (120, 60)
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        
    
    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j
        
    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z
    
    def get_pc(self):
        depth = self.obs['depth']
        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))
        
        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)
        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth
        
        pc = np.stack((xx, yy, depth, np.ones((xx.shape[0], xx.shape[1]))))  
        
        pc = pc.reshape(4, -1)
        
        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        
        rpc = np.dot(inv_E, pc).reshape(4, W, H)
        return rpc[:3, :, :]
    
    def cal_object_position(self, o_dict):        
        id_image = self.obs['seg_mask']
        depth = self.obs['depth']
        pc = self.get_pc()
        
        tttt = np.where(id_image == o_dict['seg_color'])
        
        object_pc = pc[:, tttt[0], tttt[1]].reshape(3, -1)
        object_depth = depth[tttt[0], tttt[1]].reshape(-1)
        valid = np.where(object_depth < 50)
        object_pc = object_pc[:, valid].squeeze(1)
        if object_pc.shape[1] < 30:
            
            return None
        position = object_pc.mean(1)
        return position
    
    def get_object_list(self):        
        self.visible_objects = self.obs['visible_objects']
        self.object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict['object_id'] is None:
                continue
            object_id = o_dict['object_id']
            self.object_type[object_id] = o_dict['type']
            if o_dict['type'] == 3:
                continue
            if object_id in self.finish or object_id in self.grasp:
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            self.object_position[object_id] = position
            if o_dict['type'] == 0:
                x, y, z = self.get_object_position(object_id)                
                
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 1
                    self.id_map[i, j] = object_id
                    self.object_list[0].append(object_id)
                    
            elif o_dict['type'] == 1:            
                x, y, z = self.get_object_position(object_id)
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 2
                    self.id_map[i, j] = object_id
                    self.object_list[1].append(object_id)                    
            elif o_dict['type'] == 2:            
                x, y, z = self.get_object_position(object_id)
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 3
                    self.id_map[i, j] = object_id
                    self.object_list[2].append(object_id)
        
    def is_container(self, id):
        return id is not None and id in self.object_type and self.object_type[id] == 1
        
    def container_full(self, id):
        return self.content_container > 2
    
    def have_container(self):
        self.container_held = None
        for o in self.obs["held_objects"]:
            if o is not None and self.is_container(o):
                self.container_held = o        
    
    def decide_sub(self):
        self.held_objects = self.obs["held_objects"]
        if self.num_step > self.max_steps - 150 and (self.held_objects[0] is not None or self.held_objects[1] is not None):
            self.sub_goal = 2
        else:
            self.container_held = None
            
            for o in self.held_objects:
                if self.is_container(o):
                    self.container_held = o
                    if self.container_full(o):
                        self.sub_goal = 2
                    else:
                        self.sub_goal = 0
                    return
            if self.container_held is None:
                if self.held_objects[0] is not None and self.held_objects[1] is not None:
                    self.sub_goal = 2
                elif self.num_step < self.max_steps - 200:
                    self.sub_goal = 1
                else:
                    self.sub_goal = 0
    
    def dep2map(self):  
        local_occupancy_map = np.zeros_like(self.occupancy_map, np.int32)
        local_known_map = np.zeros_like(self.occupancy_map, np.int32)
        depth = self.obs['depth']
        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))
        
        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)
        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth
        
        pc = np.stack((xx, yy, depth, np.ones((xx.shape[0], xx.shape[1]))))  
        
        pc = pc.reshape(4, -1)
        
        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        
        rpc = np.dot(inv_E, pc).reshape(4, W, H)
        
        rpc = rpc.reshape(4, -1)
        X = np.rint((rpc[0, :] - self._scene_bounds["x_min"]) / CELL_SIZE)
        X = np.maximum(X, 0)
        X = np.minimum(X, self.map_size[0] - 1)
        Z = np.rint((rpc[2, :] - self._scene_bounds["z_min"]) / CELL_SIZE)
        Z = np.maximum(Z, 0)
        Z = np.minimum(Z, self.map_size[1] - 1)
        depth = depth.reshape(-1)
        index = np.where((depth > 0.7) & (depth < 99) & (rpc[1, :] < 1.0))
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        local_known_map[XX, ZZ] = 1
        
        index = np.where((depth > 1.3) & (depth < 99) & (rpc[1, :] < -0.5))
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 0
        
        index = np.where((depth > 0.7) & (depth < 95) & (rpc[1, :] > 0.0) & (rpc[1, :] < 1.3))
        XX = X[index]
        ZZ = Z[index]
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 1
        return local_known_map
        
    def get_object_position(self, object_id):
        return self.object_position[object_id]
    
    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5
    
    def get_angle(self, forward, origin, position):        
        p0 = np.array([origin[0], origin[2]])
        p1 = np.array([position[0], position[2]])
        d = p1 - p0
        d = d / np.linalg.norm(d)
        f = np.array([forward[0], forward[2]])

        dot = f[0] * d[0] + f[1] * d[1]
        det = f[0] * d[1] - f[1] * d[0]
        angle = np.arctan2(det, dot)
        angle = np.rad2deg(angle)
        return angle

    
    def check_goal(self, thresold = 1.0):  
        if self.goal is None:
            self.d = 0
            return False
        x, _, z = self.obs["agent"][:3]
        gx, gz = self.goal
        d = self.l2_distance((x, z), (gx, gz))
        self.d = d
        return d < thresold
    
    def conv2d(self, map, kernel=3):
        from scipy.signal import convolve2d
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')
    
    def find_shortest_path(self, st, goal, map = None):
    
        st_x, _, st_z = st
        g_x, g_z = goal
        st_i, st_j = self.pos2map(st_x, st_z)
        g_i, g_j = self.pos2map(g_x, g_z)
        dist_map = np.ones_like(map, dtype=np.float32)
        super_map1 = self.conv2d(map, kernel=5)
        dist_map[super_map1 > 0] = 10
        super_map2 = self.conv2d(map)
        dist_map[super_map2 > 0] = 1000
        dist_map[map > 0] = 100000
        self.dist_map = dist_map
        
        path = pyastar.astar_path(dist_map, (st_i, st_j),
            (g_i, g_j), allow_diagonal=False)
        
        return path
    
    def Other_arm(self, arm):
        if arm == 'left':
            return 'right'
        else:
            return 'left'
    
    def ex_goal(self):
        try_step = 0
        while try_step < 7:
            try_step += 1
            goal = np.where(self.known_map == 0)
            idx = random.randint(0, goal[0].shape[0] - 1)
            i, j = goal[0][idx], goal[1][idx]
            if self.occupancy_map[i, j] == 0:
                
                self.goal = self.map2pos(i, j)
                return
        goal = np.where(self.known_map == 0)
        idx = random.randint(0, goal[0].shape[0] - 1)
        i, j = goal[0][idx], goal[1][idx]
        self.goal = self.map2pos(i, j)
        return
        
    def reset(self):
        self.is_reset = True
    
    def _reset(self):
        #self.map_size = self.info['map_size']
        self.W = self.map_size[0]
        self.H = self.map_size[1]
        #self._scene_bounds = self.info['_scene_bounds']
        
        #0: free, 1: occupied
        self.occupancy_map = np.zeros(self.map_size, np.int32)
        #0: unknown, 1: known
        self.known_map = np.zeros(self.map_size, np.int32)
        #0: free, 1: target object, 2: container, 3: goal
        self.object_map = np.zeros(self.map_size, np.int32)
        #0: unknown; object_id(only target and container)
        self.id_map = np.zeros(self.map_size, np.int32)
        
        self.static_object_info = self.info['objects_info']        
        self.object_position = {}
        self.object_type = {}
        self.sub_goal = -1
        self.mode = None    #nav, interact, move
        self.max_steps = 1000        
        self.max_nav_steps = 80
        self.max_move_steps = 150
        
        self.container_held = None
        self.grasp = {}
        self.finish = {}
        
        self.num_step = 0
        
        self.f = open('test.log', 'w')
        
        self.traj = []
        self.map_id = 0
        
        self.local_goal = None
        
        self.is_reset = False
        
    
    def pre_interact(self, object_id):
        self.mode = 'move'
        self.interact_id = object_id
        x, y, z = self.get_object_position(object_id)
        self.goal = (x, z)
        if self.sub_goal == 2:
            self.move_d = 2.5
        else:
            self.move_d = 1.5
    
    def move(self):
        self.local_step += 1
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.get_object_list()
        local_known_map = self.dep2map()
        self.known_map = np.maximum(self.known_map, local_known_map)
        path = self.find_shortest_path(self.position, self.goal, \
                                        self.occupancy_map)
        i, j = path[min(5, len(path) - 1)]
        x, z = self.map2pos(i, j)
        self.local_goal = [x, z]
        angle = self.get_angle(forward=np.array(self.forward),
                            origin=np.array(self.position),
                            position=np.array([self.local_goal[0], 0, self.local_goal[1]]))
        if np.abs(angle) < ANGLE:
            action = {"type": 0}
        elif angle > 0:
            action = {"type": 1}
        else:
            action = {"type": 2}
        
        
        return action
    
    
    def interact(self):
        if self.interact_mode == 'go_to':
            action = {"type": 3}
            action["object"] = self.interact_id
            return action
        if self.interact_mode == 'grasp':
            grasped_arm = []
            if self.obs["held_objects"][0] is None:
                grasped_arm.append('left')
            if self.obs["held_objects"][1] is None:
                grasped_arm.append('right')
            
            action = {"type": 3}
            action["object"] = self.interact_id
            action["arm"] = grasped_arm[0]
            self.grasp_arm = grasped_arm[0]
            return action
        if self.interact_mode == 'put_in':            
            action = {"type": 4,
                    "object": self.object_id,
                    "container": self.container_id}
            self.content_container += 1
            return action
        if self.interact_mode == 'drop':
            return {"type": 5}
    
    def nav(self):
        
        self.local_step += 1
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.get_object_list()
        local_known_map = self.dep2map()
        self.known_map = np.maximum(self.known_map, local_known_map)
        if len(self.object_list[1]) > 0:
            self.have_container()
            if self.sub_goal < 2 and self.container_held is None:
                self.sub_goal = 1
                goal = random.choice(self.object_list[1])                
                self.pre_interact(goal)                
                return
        if len(self.object_list[self.sub_goal]) > 0:                
            goal = random.choice(self.object_list[self.sub_goal])
            self.pre_interact(goal)            
            return
            
        path = self.find_shortest_path(self.position, self.goal, \
                                        self.occupancy_map)
        i, j = path[min(5, len(path) - 1)]
        x, z = self.map2pos(i, j)
        self.local_goal = [x, z]
        angle = self.get_angle(forward=np.array(self.forward),
                            origin=np.array(self.position),
                            position=np.array([self.local_goal[0], 0, self.local_goal[1]]))
        if np.abs(angle) < ANGLE:
            action = {"type": 0}          
        elif angle > 0:
            action = {"type": 1}             
        else:
            action = {"type": 2}

        
        return action
    
    def update_grasp(self):
        self.object_map[np.where(self.id_map == self.interact_id)] = 0
        self.id_map[np.where(self.id_map == self.interact_id)] = 0        
        self.grasp[self.interact_id] = 1
        put_arm = self.Other_arm(self.grasp_arm)
        if self.is_container(self.interact_id):
            self.object_id = None
            self.content_container = 0
            if put_arm == 'left':
                self.object_id = self.obs["held_objects"][0]
            else:
                self.object_id = self.obs["held_objects"][1]
            if self.object_id is not None:
                self.interact_mode = "put_in"
                self.container_id = self.interact_id
                return
            else:
                self.sub_goal = -1
        else:
            self.have_container()
            if self.container_held is not None:
                self.interact_mode = "put_in"
                self.container_id = self.container_held
                self.object_id = self.interact_id
                return
            else:
                self.sub_goal = -1
    
    def act(self, obs, info):
        self.obs = obs
        self.info = info
        if self.is_reset:
            self._reset()
        if self.sub_goal != -1:
            if self.mode == 'nav':
                if self.local_step >= self.max_nav_steps \
                        or self.check_goal(2.0):
                    self.sub_goal = -1                
                    
            elif self.mode == 'move':
                if self.check_goal(self.move_d):                    
                    self.mode = 'interact'
                    if self.sub_goal < 2:
                        self.interact_mode = 'grasp'#'go_to'
                        self.try_step = 0
                    else:
                        self.interact_mode = 'drop'
                elif self.local_step >= self.max_move_steps:
                    self.sub_goal = -1
                
            elif self.mode == 'interact':
                if self.interact_mode == "grasp":
                    if self.obs['status'] == True:
                        self.update_grasp()                        
                    elif self.try_step == 2:
                        self.sub_goal = -1
                    else:
                        self.try_step += 1
                else:
                    self.sub_goal = -1
        
        if self.sub_goal == -1:
            self.decide_sub()
            
            self.local_step = 0            
            goal = np.where(self.object_map == self.sub_goal + 1)
            if goal[0].shape[0] > 0:
                idx = 0
                idx = random.randint(0, len(goal[0]) - 1)
                i, j = goal[0][idx], goal[1][idx]                
                self.pre_interact(self.id_map[i, j])
            else:
                self.ex_goal()                
                self.mode = 'nav'
        self.num_step += 1
        if self.mode == 'nav':
            #self.logger.debug("Navigating ...")
            action = self.nav()
            #self.logger.debug("Got Navigating action")
            if self.mode == 'nav':
                return action
            else:
                self.local_step = 0
        if self.mode == 'move':
            #self.logger.debug("Moving ...")
            action = self.move()
            #self.logger.debug("Got Moving ...")
            return action

        if self.mode == 'interact':
            #self.logger.debug("Interacting ...")
            action = self.interact()
            self.logger.debug(action)
            #self.logger.debug("Got Interaction action")
            return action
