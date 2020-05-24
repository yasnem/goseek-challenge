from common.tools import get_config
from envs.utils import make_goseek_scene, make_navigation_scene
import os
import open3d as o3d
from common.visualization import AgentViz
from agents.policy_utils import PickupAgent

configs = get_config("debug")
training_config = configs['training']
num_env = configs['training']['num_env']
simfile = "../../simulator/goseek-v0.1.4.x86_64"
simfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), simfile)
pua = PickupAgent(configs["pickup"], configs["voxel_grid"]["camera"])
viz_geometry = {"mesh": o3d.geometry.TriangleMesh(), "tsdf": o3d.geometry.PointCloud(),
                     "grid": o3d.geometry.PointCloud(),
                     "fov_frontier": o3d.geometry.PointCloud(), "frontier": o3d.geometry.PointCloud(),
                     "agent": AgentViz().get_cylinder([0, 0, 0]), "seg": o3d.geometry.PointCloud(),
                     "target": AgentViz().get_frame([0, 0, 0])}

viz = o3d.visualization.Visualizer()
viz.create_window()


def visualize(env):
    # Update the geometries based on new information.
    global viz_geometry, viz
    pose = env.get_pose()
    viz.clear_geometries()
    viz.poll_events()
    viz.update_renderer()
    viz_geometry['tsdf'] = env.grid.feature_as_cloud(env.grid.get_tsdf())
    viz_geometry['mesh'] = env.grid.get_mesh()
    viz_geometry['frontier'] = env.grid.get_frontier()
    viz_geometry['fov_frontier'] = env.grid.get_fovfrontier(pose)
    viz_geometry['agent'] = AgentViz().get_cylinder(pose)
    viz_geometry['grid'] = env.grid.get_tsdf_voxel_grid()
    target = env.grid.get_target(pose)
    viz_geometry['seg'] = env.grid.feature_as_cloud(env.grid.get_target_likelihood())
    viz_geometry['target'] = AgentViz().get_frame(target)

    print("Current target is at ", target, "robot at ", pose)
    for k in ["mesh", "agent", "target", 'frontier', "fov_frontier"]:
        viz.add_geometry(viz_geometry[k])
    ctr = viz.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
    ctr.convert_from_pinhole_camera_parameters(param)
    for k in ["mesh", "agent", "target", 'frontier', "fov_frontier"]:
        viz.update_geometry(viz_geometry[k])
    print("Current fov frontier size: ", len(viz_geometry['fov_frontier'].points))
    viz.poll_events()
    viz.update_renderer()

if training_config['task'] == 'goseek':
    env = make_goseek_scene(scene=training_config['scene_id'][0],
                            filename=simfile,
                            reward_config=configs['rewards'],
                            grid_config=configs['voxel_grid'],
                            restart_on_collision=training_config['restart_on_collision'],
                            n_targets=training_config['targets_num'],
                            episode_length=training_config['ep_len'])
elif training_config['task'] == 'navigation':
    env = make_navigation_scene(scene=training_config['scene_id'][0],
                                filename=simfile,
                                reward_config=configs['rewards'],
                                grid_config=configs['voxel_grid'],
                                episode_length=training_config['ep_len'])

env.reset()
visualize(env)

while True:  # making a loop
    done = False
    [_, seg, depth] = env.observe().images
    if pua.decide_pickup(seg, depth):
        print("The PUA think you should do it. ")
    pressed = input("use w for up, a for left, d for right, s for collect, space reset and q to quit:\n")
    if pressed == 'w':  # if key 'q' is pressed
        print('Robot moving forward.')
        obs, reward, done, infos = env.step(0)
        visualize(env)
    elif pressed == 'a': # if key 'a' is pressed
        print('Robot turning to left.')
        obs, reward, done, infos = env.step(2)
        visualize(env)
    elif pressed=='d': # d is pressed
        print("robot turning to right")
        obs, reward, done, infos = env.step(1)
        visualize(env)
    elif pressed=='s': # s is pressed
        print("robot picks up")
        obs, reward, done, infos = env.step(3)
        print(infos)
        visualize(env)
    elif pressed == ' ':
        print("env restart requested. ")
        env.reset()
        visualize(env)
    elif pressed == 'q':
        print("quit")
        break

    if done:
        ob = env.reset()
        visualize(env)

viz.destroy_window()