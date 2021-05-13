import os
import json
import gym
import time
import pickle
from pkg_resources import resource_filename

import tdw_transport_challenge
import pkg_resources

class Challenge:
    def __init__(self, logger, port = 1071):
        self.env = gym.make("transport_challenge-v0", \
                        train = 1, \
                        physics = True, \
                        ip_address=None, \
                        port = port)
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.max_step_num = 1000
        # Check if custom pkl file is given
        if os.path.isfile("/test_dataset/test_dataset.pkl"):
            with open(pkg_resources.resource_filename("tdw_transport_challenge", 'test_dataset.pkl'), 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(pkg_resources.resource_filename("tdw_transport_challenge", 'train_dataset.pkl'), 'rb') as f:
                self.data = pickle.load(f)


    def submit(self, agent):
        total_reward = 0.0
        total_finish = 0.0
        if "NUM_EVAL_EPISODES" in os.environ:
            num_eval_episodes = int(os.environ["NUM_EVAL_EPISODES"])
        else:
            num_eval_episodes = 4   #for test env
        
        start = time.time()
        for i in range(0, num_eval_episodes):
            
            self.logger.info('Episode: {}/{}'.format(i + 1, num_eval_episodes))
            try:
                agent.reset()
            except:
                pass
            start_time = time.time()
            self.logger.debug("Resetting Environment ...")
            state, info = self.env.reset(self.data[i])
            #for debug
            self.env.save_images()
            
            self.logger.debug(f'position: {info["position"]}')
            self.logger.debug(f"Environment Reset !! Took {time.time() - start_time} secs")
            step_num = 0
            local_reward = 0
            local_finish = 0
            while step_num <= self.max_step_num:                
                self.logger.info(f"Executing step {step_num} for episode: {i}, reward: {local_reward}, finish: {local_finish}")
                step_num += 1
                action = agent.act(state, info)
                state, reward, done, info = self.env.step(action)
                local_reward += reward
                local_finish = info['finish']                
                if done:
                    break
            total_finish += local_finish
            total_reward += local_reward
        avg_reward = total_reward / num_eval_episodes
        avg_finish = total_finish / num_eval_episodes
        results = {
            "avg_reward": avg_reward
        }
        if os.path.exists('/results'):
            with open('/results/eval_result.json', 'w') as f:
                json.dump(results, f)
        self.logger.info('eval done, avg reward {}, avg_finish {}'.format(avg_reward, avg_finish))
        self.logger.info('time: {}'.format(time.time() - start))
        return total_reward

    def close(self):
        self.env.close()
        
    
if __name__ == "__main__":
    challenge = Challenge()
