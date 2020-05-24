import time
import os

from envs.utils import make_navigation_scene, make_goseek_scene
from agents.agent import SACAgent
import argparse
from common.tools import get_config, get_config_file
from shutil import copyfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', '-env', type=str,
                        default="../../simulator/goseek-v0.1.4.x86_64")
    parser.add_argument('--config', type=str, default='navigation')
    # Visualizing that the feature extraction is correct during simulation.
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    configs = get_config(params['config'])
    print("Using env configuration: ", configs['training']['task'])

    # Setup visualization in the config.
    if params['debug']:
        configs['voxel_grid']['debug'] = True
        print("Entering debug mode for visualizing all extracted features and show rewards.")

    ##################################
    # CREATE DIRECTORY FOR LOGGING
    ##################################
    logdir_prefix = configs['training']['task'] + '_' + configs['agent']['type'] + "_"
    # directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../logs')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + configs['training']['exp_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    copyfile(get_config_file(params['config']), logdir+"/config.yaml")

    ###################
    # Simulation ENV
    ###################

    simfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['env_file'])
    print("Loading from sim file ", simfile)
    training_config = configs['training']
    num_env = configs['training']['num_env']
    if num_env == 1:
        if training_config['task'] == 'goseek':
            env = make_goseek_scene(scene=training_config['scene_id'][0],
                                    filename=simfile,
                                    reward_config=configs['rewards'],
                                    grid_config=configs['voxel_grid'],
                                    auto_pickup=training_config['auto_pickup'],
                                    restart_on_collision=training_config['restart_on_collision'],
                                    n_targets=training_config['targets_num'],
                                    episode_length=training_config['ep_len'])
        elif training_config['task'] == 'navigation':
            env = make_navigation_scene(scene=training_config['scene_id'][0],
                                        filename=simfile,
                                        reward_config=configs['rewards'],
                                        grid_config=configs['voxel_grid'],
                                        episode_length=training_config['ep_len'])

    else:
        raise NotImplementedError("Currently stacked env doesn't work.")

    ###################
    # Set up rl agent
    ###################
    agent_config = configs['agent']
    if agent_config['type'] == 'sac':
        agent = SACAgent(env, logdir, agent_config)
    else:
        raise NotImplementedError("Currently only support sac algorithm. ")


    ###################
    # RUN TRAINING
    ###################

    agent.train(n_timesteps=training_config['timestamps'],
                log_interval=training_config['log_interval'])
    # shut down the environment.
    env.close()
    agent.save()


if __name__ == "__main__":
    main()
