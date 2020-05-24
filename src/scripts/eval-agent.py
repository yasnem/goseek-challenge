import argparse
from envs.utils import make_goseek_scene, make_navigation_scene
from agents.agent import SACAgent
from common.tools import get_config
from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', '-env', type=str,
                        default="../../simulator/goseek-v0.1.4.x86_64")
    parser.add_argument('--weights_file', '-w', type=str, required=True)
    parser.add_argument('--config', type=str, default='navigation')
    # parse arguments
    args = parser.parse_args()
    params = vars(args)
    configs = get_config(params['config'])
    eval_config = configs['evaluation']
    training_config = configs['training']

    ###################
    # Load Model      #
    ###################
    agent = SACAgent(env=None, log_dir=None,
                     config=configs['agent'], 
                     weights=params['weights_file'])

    ###################
    # Simulation ENV  #
    ###################
    reward_stats = []
    for scene in eval_config['scene_ids']:

        if eval_config['task'] == 'navigation':
            env = make_navigation_scene(scene=scene,
                                        filename=params['env_file'],
                                        reward_config=configs['rewards'],
                                        grid_config=configs['voxel_grid'],
                                        episode_length=training_config['ep_len'])
        else:
            env = make_goseek_scene(scene=scene,
                                    filename=params['env_file'],
                                    reward_config=configs['rewards'],
                                    grid_config=configs['voxel_grid'],
                                    restart_on_collision=training_config[
                                        'restart_on_collision'],
                                    n_targets=training_config['targets_num'],
                                    episode_length=training_config['ep_len'])

        mean_reward, std_reward = evaluate_policy(agent.model, env,
                                                  n_eval_episodes=eval_config['n_episodes'])
        env.close()
        reward_stats.append((mean_reward, std_reward))

    for idx, scene in enumerate(eval_config['scene_ids']):
        print("Environment scene {0}: Mean reward {1} +- Std_reward {"
              "2}".format(scene, reward_stats[idx][0], reward_stats[idx][1]))


if __name__ == "__main__":
    main()
