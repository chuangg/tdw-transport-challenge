import numpy as np
import random

import pickle

if __name__ == '__main__':
    train = 1
    if train == 0:
        scenes = ['5a', '5b', '2a', '2b', '1a']
        layouts = [0, 1]
        l = 100
        path = 'train_dataset.pkl'
    elif train == 1:   
        scenes = ['5a', '5b', '2a', '2b', '5c']
        layouts = [2]        
        l = 20
        path = 'test_dataset.pkl'
    else:
        scenes = ['5a', '2a']
        layouts = [0]
        l = 2
        path = 'test_env.pkl'
    print(path)
    scene_info = []
    for scene in scenes:
        for layout in layouts:
            seeds = []
            for i in range(l):
                seed = random.randint(0, 10000)
                while seed in seeds:
                    seed = random.randint(0, 10000)
                seeds.append(seed)
                scene_info.append({
                    'scene': scene, \
                    'layout': layout, \
                    'seed': seed
                })
    with open(path, 'wb') as f:
        pickle.dump(scene_info, f)