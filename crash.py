from transport_challenge import Transport
import pickle
import gym
import requests
import json
import random
def create_tdw(port):
    url = "http://localhost:5000/get_tdw"
    data = {
        'ip_address': "localhost",
        'port': port
    }
    response = requests.post(url, json=json.dumps(data))
    print(response.status_code, response.reason)
    docker_id = response.json()['docker_id']
    return docker_id


def kill_tdw(docker_id):
    url = "http://localhost:5000/kill_tdw"
    data = {
        "container_id": docker_id
    }
    response = requests.post(url, json=json.dumps(data))
    print(response.status_code, response.reason)


if __name__ == '__main__':
    port = 1090
    docker_id = create_tdw(port = port)
    try:
        m = Transport(port=port, launch_build=False, \
                screen_width=128, screen_height=128, fov = 90)
        #path = './to_send/tdw_crash/submissions_4/actions_history.pickle'
        path = 'action.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)     
        
        n = len(data)
        
        print('done')
        scene_info, actions = data['scene_info'], data['actions']
        m.init_scene(scene = scene_info['scene'], \
                        layout = scene_info['layout'], \
                        random_seed = scene_info['seed'])
        m.communicate([{"$type": "set_floorplan_roof", "show": False}])
        x, y, z = m.state.magnebot_transform.position
        print('init:', m.state.magnebot_transform.position, m.state.magnebot_transform.forward)
        m.turn_by(angle=-15, aligned_at=2)
        print('second:', m.state.magnebot_transform.position, m.state.magnebot_transform.forward)
        assert False
        #m.add_camera({"x": x, "y": 5, "z": z}, look_at=True, follow=True)
        i = 0
        for action in actions:
            i += 1
            if action['type'] == 0:
                status = m.move_by(0.5, arrived_at = 0.1)
                
            elif action['type'] == 1:
                status = m.turn_by(angle=-15, aligned_at=2)
            elif action['type'] == 2:
                status = m.turn_by(angle=15, aligned_at=2)
            #elif action['type'] == 3:
            #    m.move_to(int(action['object']))
            print(f'execute step: {i}, action: {action["type"]}, status: {status}')
        '''for episode in data.keys():
            scene_info, actions = data[episode]['seed'], data[episode]['actions']        
            m.init_scene(scene = scene_info['scene'], \
                            layout = scene_info['layout'], \
                            random_seed = scene_info['seed'])
            i = 0
            print('episode: {}, length: {}'.format(episode, len(actions)))
            for action in actions:
                i += 1
                if action['type'] == 0:
                    status = m.move_by(0.5, arrived_at = 0.1)
                    
                elif action['type'] == 1:
                    status = m.turn_by(angle=-15, aligned_at=2)
                else:
                    status = m.turn_by(angle=15, aligned_at=2)
                '''
    finally:
        kill_tdw(docker_id)
        