from tesse_gym.tasks.goseek.goseek_full_perception import GoSeekFullPerception
from tesse_gym.tasks.navigation.navigation import Navigation
from tesse_gym import get_network_config
from stable_baselines3.common.vec_env import SubprocVecEnv
from tesse_gym.core.utils import set_all_camera_params
from envs.tsdf_wrapper import GoseekTSDF, NavigationTSDF
from common.tools import get_rewards_config


scene_id = [1, 2, 3, 4, 5]  # list all available scenes


def make_unity_env(num_env, task, filename, reward_config, n_targets=30, episode_length=400):
    """ Create a wrapped Unity environment. """

    def make_env(rank):
        def _thunk():
            if task == 'goseek':
                env = GoSeekFullPerception(
                    str(filename),
                    network_config=get_network_config(worker_id=rank),
                    n_targets=n_targets,
                    episode_length=episode_length,
                    scene_id=scene_id[rank],
                    step_rate=20,
                    init_hook=set_all_camera_params
                )
            else:
                env = Navigation(str(filename),
                                 network_config=get_network_config(
                                     worker_id=rank),
                                 rewards=reward_config,
                                 episode_length=episode_length,
                                 scene_id=scene_id[rank])
            return env

        return _thunk

    return SubprocVecEnv([make_env(i) for i in num_env])


def make_goseek_scene(scene, filename, reward_config, grid_config, auto_pickup=False, restart_on_collision=False, n_targets=30, episode_length=400):
    """Create single full perception environment."""
    assert scene in scene_id
    worker_id = scene_id.index(scene)
    return GoseekTSDF(str(filename),
                      network_config=get_network_config(worker_id=(worker_id-1)),
                      n_targets=n_targets,
                      episode_length=episode_length,
                      scene_id=scene,
                      rewards=reward_config,
                      step_rate=20,
                      init_hook=set_all_camera_params,
                      restart_on_collision=restart_on_collision,
                      grid=grid_config)


def make_navigation_scene(scene, filename, reward_config, grid_config, episode_length=400):
    """Create single navigation environment."""
    assert scene in scene_id
    worker_id = scene_id.index(scene)

    return NavigationTSDF(str(filename),
                          network_config=get_network_config(worker_id=(worker_id-1)),
                          episode_length=episode_length,
                          scene_id=scene,
                          rewards=reward_config,
                          step_rate=20,
                          grid=grid_config)

# parameters:
# https://github.com/MIT-TESSE/tesse-gym/blob/402fc85d39b2075c37763f7aa92aa2860663a54c/src/tesse_gym/tasks/goseek/goseek.py#L56
