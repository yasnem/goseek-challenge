import warnings
warnings.filterwarnings('ignore')
from tesse_gym import get_network_config
from tesse_gym.core.utils import set_all_camera_params
from tesse_gym.tasks.goseek.goseek_full_perception import decode_observations
from envs.utils import make_ith_scene
from common.tools import convert_pose_stack
import numpy as np
import tqdm
import os

### TODO(JD): THis is currently unmaintained

poses = []
ind = 0
env = make_ith_scene(1)
episode_length = 10
test_data_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/../test/data/'
# Save observations to desired folder
def saveCurrent():
    global ind, poses
    observed = env.observe()
    rgb, segmentation, depth= observed.images
    np.save(test_data_folder + 'color_{:05d}'.format(ind), rgb)
    np.save(test_data_folder + 'segment_{:05d}'.format(ind), segmentation)
    np.save(test_data_folder + 'depth_{:05d}'.format(ind), depth)
    env._update_pose(observed.metadata)
    pose = env.get_pose()
    poses.append(pose)
    ind += 1

# Using a random agent.
action_space = np.arange(0, 4)
action_probability = np.array([0.3, 0.3, 0.3, 0])
for i in tqdm.tqdm(range(episode_length)):
    # give probability for actions in `action_space` front, left, right, pickup.
    action = np.random.choice(action_space, p=action_probability)
    obs, reward, done, pose = env.step(action)
    saveCurrent()
    poses.append(pose)

converted = convert_pose_stack(poses)
ind1, ind2 = 0, 1
# Save pose in desired format
with open(test_data_folder+'/traj.log', 'w') as f:
    for p in converted:
        f.write(str(ind1) + ' ' +  str(ind1) + ' ' + str(ind2) + '\n')
        ind1 += 1
        ind2 += 1
        for x in p:
            f.write(' '.join(str(x[i]) for i in range(4)) + '\n')