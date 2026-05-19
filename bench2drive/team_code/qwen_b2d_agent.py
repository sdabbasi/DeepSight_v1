import os
import re
import cv2
import time
import json
import math
import copy
import carla
import torch
import datetime
import pathlib
import numpy as np
from PIL import Image
from scipy.optimize import fsolve
from scipy.interpolate import splprep, splev
from team_code.pid_controller import PIDController
from team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


# os.environ['SAVE_PATH'] = "eval_output/debug"
# os.environ['ROUTES'] = "leaderboard/data/bench2drive220"
SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
# IS_BENCH2DRIVE = True


def calculate_cube_vertices(center, extent):
    if isinstance(center, list):
        cx, cy, cz = center
        x, y, z = extent
    else:
        cx, cy, cz = center.x,  center.y,  center.z
        x, y, z = extent.x, extent.y, extent.z
    vertices = [
        (cx + x, cy + y, cz + z),
        (cx + x, cy + y, cz - z),
        (cx + x, cy - y, cz + z),
        (cx + x, cy - y, cz - z),
        (cx - x, cy + y, cz + z),
        (cx - x, cy + y, cz - z),
        (cx - x, cy - y, cz + z),
        (cx - x, cy - y, cz - z)
    ]
    return vertices

def float_to_uint8_color(float_clr):
    assert all([c >= 0. for c in float_clr])
    assert all([c <= 1. for c in float_clr])
    return [int(c * 255.) for c in float_clr]


def get_entry_point():
    return 'QwenAgent'

# 初始化模型
def init_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    return processor, model, tokenizer


def format_trajs(trajs):
    str_trajs = []
    for traj in trajs:
        x, y = traj
        x = f'{x:.2f}'
        y = f'{y:.2f}'
        if x == '-0.00':
            x = '0.00'
        if y == '-0.00':
            y = '0.00'
        traj = f'({x},{y})'
        str_trajs.append(traj)
    str_trajs = ','.join(str_trajs)
    return f'[{str_trajs}]'


def get_prompt(command, his_trajs=None, speed_content=None, bevtargetpoints=None):
    history_prompt = "These are the vehicle's CAM_FRONT historical images: 2.0s ago <image> 1.5s ago <image> 1.0s ago <image> 0.5s ago <image>."
    soround_prompt = "These are the vehicle's current frame six-view images: CAM_FRONT:<image> CAM_FRONT_LEFT:<image> CAM_FRONT_RIGHT:<image> CAM_BACK:<image> CAM_BACK_LEFT:<image> CAM_BACK_RIGHT:<image>."
    state_promt = f"These are the target pixel tokens: {bevtargetpoints} Historical trajectory: {his_trajs} current speed info: {speed_content}"
    instruct_promt1 = "<CoT_flag_True>"
    instruct_promt = "Based on the provided particulars, please generate BEV image and plan waypoints (0.5s intervals) for the next 2 seconds.\n"
    prompt = '\n'.join([history_prompt, soround_prompt, state_promt, instruct_promt1,instruct_promt])
    return prompt


def get_images(i, history_path, suround_view_path):
    images = []
    for index in range(4, 0, -1):
        his_index = i - index * 5
        if his_index < 0:
            img_file = '/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/sft_data_api_explain/hisblack.jpg'
        else:
            img_file = os.path.join(history_path, f'{his_index:05d}.jpg')
        assert os.path.exists(img_file), f"{img_file} not exists"
        images.append(img_file)
    # 环视图图像
    for cam_path in suround_view_path:
        img_file = os.path.join(cam_path, f'{i:05d}.jpg')
        assert os.path.exists(img_file), f"{img_file} not exists"
        images.append(img_file)

    assert len(images) == 10, f"{suround_view_path} {i} {len(images)}"
    return images


class QwenAgent(autonomous_agent.AutonomousAgent):

    def __init__(self, host, port, debug):
        super(QwenAgent, self).__init__(host, port, debug)
        
        #======== only for use map info ========#
        self.town_name : str = ""
        self.lat_ref : float = 42.0
        self.lon_ref : float = 2.0
        self.his_trajs = []
        self.his_images = []
        self.frame = 10 # 20

    def setup(self, model_path, keypoints, manager=None):
        self.manager = manager
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        self.pidcontroller = PIDController() 
        self.waypoints = self.get_sampled_points(keypoints)
        # print(self.waypoints)
        
        if IS_BENCH2DRIVE:
            self.save_name = model_path.split('+')[-1]
        else:
            now = datetime.datetime.now()
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        
        self.model_path = model_path.split('+')[0]
        self.torch_dtype = "bf16"

        # print('使用原始代码进行推理, 开始初始化模型')
        self.processor, self.model, self.tokenizer = init_model(self.model_path)
        self.model.eval()

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None

        #======== comment out only for use map info ========#
        # self.lat_ref, self.lon_ref = 42.0, 2.0
        #==================================================#

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / pathlib.Path("Scenarios") / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'anno').mkdir()
            (self.save_path / 'bev').mkdir()
            for cam_key in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT', 'CAM_BEV']:
                cam_path = os.path.join(self.save_path, 'camera', cam_key)
                os.makedirs(cam_path, exist_ok=True)
   
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -1.  , -0.24],
                              [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        # topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], 
        #                                 [0.0,  1.0, -0.0,  0.0], 
        #                                 [1.0, -0.0,  0.0, -0.0], 
        #                                 [0.0,  0.0,  0.0,  1.0]])
        # unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        #unreal2cam @ topdown_extrinsics
        self.coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0], 
                                      [0.0, -1.0,  0.0,  0.0], 
                                      [0.0,  0.0, -1.0, 50.0], 
                                      [0.0,  0.0,  0.0,  1.0]])
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 512.0, 0], [0.0, 548.993771650447, 512.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def _init(self, input_data):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0      
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

        # gps = input_data['GPS'][1][:2]
        # pos = self.gps_to_location(gps)
        # x, y = pos[0], -pos[1]
        # past_frames = (self.cfg.history_frames)*10+1  # 20 Hz
        # self.history_states = np.array([[x, y]]*past_frames)

        self.throttle = 0
        self.brake = 0
        self.steer = 0

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
            ]
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 1600, 'height': 900, 'fov': 110,
                        'id': 'CAM_BEV'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT', 'CAM_BEV']:
            img = input_data[cam][1][:, :, :3]
            imgs[cam] = img
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
  
        if math.isnan(compass): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        
        bounding_boxes = self.get_bounding_boxes()

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node.tolist(),
                'bounding_boxes': bounding_boxes
                }
        
        return result
    
    def format_message(self, tick_data):
        command = tick_data['command_near']
        speed_content = f'speed: {tick_data["speed"]:.2f}, acceleration: {tick_data["acceleration"][0]:.2f}'
        his_trajs = []
        his_images = []
        for i in range(4, 0, -1):
            his_index = i * self.frame // 2
            if len(self.his_trajs) < his_index:
                if len(self.his_trajs) == 0:
                    his_traj = tick_data['bounding_boxes']['location']
                else:
                    his_traj = copy.deepcopy(self.his_trajs[0])
                his_images.append('/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/sft_data_api_explain/hisblack.jpg')
            else:
                his_traj = self.his_trajs[-his_index]
                his_images.append(self.his_images[-his_index])
            his_trajs.append(his_traj)
        world2ego = np.array(tick_data['bounding_boxes']['world2ego'])
        targetpoint = self.waypoints
        extent_z = tick_data['bounding_boxes']['extent'][2]
        ego_location_z = tick_data['bounding_boxes']['location'][2]


        # sensor_config = {
        # 'x': 0.0, 'y': 0.0, 'z': 50.0,
        # 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0
        # }
        # location1 = carla.Location(x=sensor_config['x'], y=sensor_config['y'], z=sensor_config['z'])
        # rotation1 = carla.Rotation(roll=sensor_config['roll'], pitch=sensor_config['pitch'], yaw=sensor_config['yaw'])
        # transform1 = carla.Transform(location1, rotation1)
        # cam2ego = np.array(transform1.get_matrix())
        cam2ego = np.array([
            [6.123233995736766e-17, -0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, -0.0, 6.123233995736766e-17, 50.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        ego2cam = np.linalg.inv(cam2ego)
        world2cam = np.dot(ego2cam, world2ego)
        intrinsic_topdownd=[
                [
                    560.1660305677678,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    560.1660305677678,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        intrinsic_topdownd = np.array(intrinsic_topdownd)
        bevtargetpoints = []
        for point in targetpoint:
            points_3d = [point[0],point[1], ego_location_z - extent_z]
            points_4d = np.array(points_3d + [1])
            bevtarget = world2cam @ points_4d
            Zc, Xc, Yc = bevtarget[:3]
            fx, fy = intrinsic_topdownd[0][0], intrinsic_topdownd[1][1]
            cx, cy = intrinsic_topdownd[0][2], intrinsic_topdownd[1][2]
            u = fx * (Xc / Zc) + cx
            v = - fy * (Yc / Zc) + cy
            dx, dy = round((u - 800)/2), round((450 - v)/2)
            # dx, dy = max(-255, min(255, dx)), max(-255, min(255, dy))
            if dy < -255 or dy > 255 or dx < -255 or dx > 255:
                continue  # 跳过当前时间点，不添加到结果
            dx, dy = f'<|pixel_token_{dx}|>',f'<|pixel_token_{dy}|>'
            bevtargetpoints.append(f'({dy},{dx})')
        bevtargetpoints = ','.join(bevtargetpoints)
        bevtargetpoints = f'[{bevtargetpoints}]'
        # print(bevtargetpoints)
        # 1218 change get pixel value in bev images

        # world2cam = np.array(all_annos[index]["sensors"]["TOP_DOWN"]['world2cam'])
        # intrinsic = np.array(all_annos[index]['sensors']["TOP_DOWN"]["intrinsic"])
        ego_his_trajs = []
        for his_traj in his_trajs:
            his_traj = world2ego @ np.array(his_traj + [1])
            his_traj = tuple(his_traj[:2].tolist())
            ego_his_trajs.append(his_traj)
            
        his_trajs = format_trajs(ego_his_trajs)

        prompt = get_prompt(command=command, his_trajs=his_trajs, speed_content=speed_content, bevtargetpoints=bevtargetpoints)     # 获取 prompt

        # 环视图图像
        img_keys = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        images = [str(self.save_path / 'camera' / f'{key}' / (f'{self.step:05}.jpg')) for key in img_keys]
        images = his_images + images

        return prompt, images, bevtargetpoints
    

    def sample_path_equidistant(self, waypoints, interval=5.0):
        """
        从起点开始沿路径进行等距离采样
        
        参数:
        - waypoints: 路径点列表 [(x1,y1), (x2,y2), ...]
        - interval: 采样间隔（L2距离单位）
        
        返回:
        - 采样点列表 [(x1,y1), (x2,y2), ...]
        """
        if len(waypoints) < 2:
            return waypoints  # 不足两点直接返回
        
        samples = [waypoints[0]]  # 起点
        current_dist = 0.0
        next_sample_dist = interval
        
        for i in range(1, len(waypoints)):
            p0 = np.array(waypoints[i-1])
            p1 = np.array(waypoints[i])
            seg_len = np.linalg.norm(p1 - p0)
            
            # 在当前线段上生成所有采样点
            while next_sample_dist <= current_dist + seg_len:
                ratio = (next_sample_dist - current_dist) / seg_len
                sample_point = p0 + ratio * (p1 - p0)
                samples.append(tuple(sample_point))
                next_sample_dist += interval
            
            current_dist += seg_len
            
        if len(samples) > 0 and waypoints[-1] != samples[-1]:
            samples.append(waypoints[-1])
        
        return samples

    def calculate_angle(self, v1, v2):
        """计算两个向量之间的夹角（度）"""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0
        
        cos_theta = dot / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle

    def line_intersection(self, line1, line2):
        """计算两条直线的交点"""
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if abs(div) < 1e-8:
            return None  # 平行线

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return (x, y)

    def crop_sharp_turns_improved(self, waypoints, angle_threshold=5, max_extension=150.0):
        """
        使用前后两个点定义的直线交点裁剪急转弯点，并避免重复处理参与计算的点
        
        参数:
        - waypoints: 原始轨迹点列表
        - angle_threshold: 角度阈值（度），超过此值视为急转弯
        - max_extension: 裁剪点与原点的最大允许距离
        
        返回:
        - 裁剪后的轨迹点列表
        """
        if len(waypoints) < 5:  # 至少需要5个点才能获取前后各两个点
            return waypoints
        
        points = np.array(waypoints, dtype=float)
        cropped = [tuple(points[0])]  # 保留起点
        
        # 处理第二个点
        cropped.append(tuple(points[1]))
        
        i = 2  # 从第三个点开始处理
        while i < len(points) - 2:  # 直到倒数第三个点结束
            # 计算前后向量
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            
            # 计算夹角
            angle = self.calculate_angle(v1, v2)
            
            if angle > angle_threshold:
                # 定义前面的直线：使用 (i-2, i-1) 这两个点
                line1 = (tuple(points[i-2]), tuple(points[i-1]))
                
                # 定义后面的直线：使用 (i+1, i+2) 这两个点
                line2 = (tuple(points[i+1]), tuple(points[i+2]))
                
                # 计算两条直线的交点
                intersection = self.line_intersection(line1, line2)
                
                if intersection:
                    # 检查交点是否合理（距离不能太远）
                    dist_to_turn = np.linalg.norm(np.array(intersection) - points[i])
                    if dist_to_turn < max_extension:
                        # 用交点替代当前转弯点
                        cropped.append(intersection)
                        
                        # 关键改进：跳过被处理过的点
                        # 跳过当前点和后面两个参与计算的点
                        i += 3  # 直接跳到 i+3
                        continue
            
            # 没有急转弯或交点无效，保留当前点
            cropped.append(tuple(points[i]))
            i += 1  # 正常递增
        
        # 添加剩余未处理的点
        for j in range(max(i, len(points)-2), len(points)):
            cropped.append(tuple(points[j]))
        
        # 移除重复点
        unique_points = []
        for p in cropped:
            if not unique_points or np.linalg.norm(np.array(p) - np.array(unique_points[-1])) > 1e-5:
                unique_points.append(p)
        
        return unique_points

    def get_sampled_points(self, waypoints, angle_threshold=15, max_extension=5000.0):
        """
        仅绘制等距离采样点（L2距离=5）
        
        参数:
        - waypoints: 原始轨迹点列表
        - color: 采样点颜色
        - angle_threshold: 角度阈值（度）
        - max_extension: 裁剪点与原点的最大允许距离
        
        返回:
        - 绘制的散点对象
        """
        if len(waypoints) < 2:
            return None
        
        # 1. 移除重复点
        filtered = []
        last = None
        for p in waypoints:
            if last is None or np.linalg.norm(np.array(p) - np.array(last)) > 1e-5:
                filtered.append(p)
                last = p
        
        if len(filtered) < 2:
            return None
        
        # 2. 应用急转弯裁剪
        cropped = self.crop_sharp_turns_improved(filtered, angle_threshold, max_extension)
        
        # 3. 等距离采样 (L2距离=5)
        samples = self.sample_path_equidistant(cropped, interval=10)
        
        return samples
    
    def add_bev_text(self, text):
        t, h, w, patchsize, n_cls, n_register = 5, 256, 256, 16, 1, 4
        l = t * (h * w // (patchsize ** 2) + n_cls + n_register)
        bev_content = []
        for i in range(l):
            bev_content.append(f"<|bev_token_{i}|>")
        bev_content = ''.join(bev_content)
        text = text + '<|start_bev_token|>' + bev_content + '<|end_bev_token|>\n'
        return text

    def data_process(self, message, images):
        content = message.split('<image>')
        assert len(content) == 11
        format_content = []
        for i in range(len(content)):
            c = {
                "type": "text",
                "text": content[i]
            }
            format_content.append(c)
            if i != len(content) - 1:
                c = {
                    "type": "image",
                    "image": images[i],
                    "resized_height":364, "resized_width":644
                }
                format_content.append(c)
        messages = [ { "role": "user",  "content": format_content }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 他的作用是？
        text = self.add_bev_text(text)
        print(messages)
        # print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=text, images=[image_inputs], videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        return inputs
    
    def decode_traj(self, inputs, generated_ids):
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        # text
        output_text = self.tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
        # parse traj
        # future_trajs_pixel = output_text.split('future pixel tokens: ')[1].split('. </answer>')[0]
        future_trajs = output_text.split('future waypoints: ')[1].split('. </answer>')[0]
        # 轨迹转坐标
        pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
        matches = re.findall(pattern, future_trajs)
        future_trajs = [[float(matches[i+1]), float(matches[i])] for i in range(0, len(matches), 2)]
        future_trajs = np.array(future_trajs)
        return future_trajs, output_text

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        # 更新carla原始输入数据
        if not self.initialized:
            self._init(input_data)
            tick_data = self.tick(input_data)
        else:
            tick_data = self.tick(input_data)
        # 保存当前帧数据
        self.save_cur_frame(tick_data)
        # 更改数据格式
        message, images, bevpixel = self.format_message(tick_data)
        # 进行数据处理 processor
        inputs = self.data_process(message, images)
        # model forward
        anwser_ids = self.model.generate(**inputs, max_new_tokens=15000)
        # decode 2 traj
        planning_traj, anwser = self.decode_traj(inputs, anwser_ids) # np.array (4, 2)
        print(anwser)
        # pid control
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(planning_traj, tick_data['speed'])

        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)

        self.throttle = control.throttle
        self.brake = control.brake
        self.steer = control.steer

        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['prompt'] = message
        self.pid_metadata['output'] = anwser
        # self.pid_metadata['local_command_xy'] = results['local_command_xy'].tolist()
        self.pid_metadata['command_near'] = tick_data['command_near']
        self.pid_metadata['command_near_xy'] = tick_data['command_near_xy']
        self.pid_metadata['bounding_boxes'] = tick_data['bounding_boxes']
        self.pid_metadata['timestamp'] = timestamp
        self.pid_metadata['planning_trajectory'] = round_two_dim_list(planning_traj.tolist())
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        self.bevpixel = bevpixel

        if SAVE_PATH is not None and self.step % 1 == 0:
            self.save(tick_data)
        self.prev_control = control
        
        return control

    def save(self, tick_data):
        frame = self.step
        self.his_trajs.append(tick_data['bounding_boxes']['location'])
        cam_front_img = str(self.save_path / 'camera' / 'CAM_FRONT' / (f'{frame:05}.jpg'))
        assert os.path.exists(cam_front_img)
        self.his_images.append(cam_front_img)  
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w', encoding='utf-8')
        json.dump(
            self.pid_metadata, 
            outfile, 
            indent=4,
            ensure_ascii=False  # 关键修复！
        )
        outfile.close()
        # metric
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()
        imgs_with_box = {}
        imgs_with_box['bev'] = self.draw_traj_bev(self.bevpixel, tick_data['imgs']['CAM_BEV'])
        for cam, img in imgs_with_box.items():
            Image.fromarray(img).save(self.save_path / str.lower(cam).replace('cam','rgb') / ('%04d.png' % frame))   
        
    def save_cur_frame(self, tick_data):
        images = tick_data['imgs']
        frame = self.step
        for key, img in images.items():
            img_path = str(self.save_path / 'camera' / (key) / (f'{frame:05}.jpg'))
            # cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 20])

    def draw_traj(self, traj, raw_img,canvas_size=(900,1600),thickness=3,is_ego=True,hue_start=120,hue_end=80):
        line = traj
        lidar2img_rt = self.lidar2img['CAM_FRONT']
        img = raw_img.copy()
        pts_4d = np.stack([line[:,0],line[:,1],np.ones((line.shape[0]))*(-1.84),np.ones((line.shape[0]))])
        pts_2d = ((lidar2img_rt @ pts_4d).T)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
        if not mask.any():
            return img
        pts_2d = pts_2d[mask,0:2]
        if is_ego:
            pts_2d = np.concatenate([np.array([[800,900]]),pts_2d],axis=0)
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except:
            return img
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
        
        num_points = len(smoothed_pts)
        for i in range(num_points-1):
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
            cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)  
      
        return img

    def transform_pixel2pixel(self, trajs_pixel):
    # 相对坐标系转绝对坐标系
        for traj in trajs_pixel:
            traj[0] = 800 + traj[0]*2
            traj[1] = 450 - traj[1]*2

        return trajs_pixel

    def draw_traj_bev(self, traj, raw_img,canvas_size=(1024,1024),thickness=3,is_ego=False,hue_start=120,hue_end=80):
        pattern = r'<\|pixel_token_([-+]?\d+)\|>'
        matches = re.findall(pattern, traj)
        targetpointpixel = [[int(matches[i+1]), int(matches[i])] for i in range(0, len(matches), 2)]
        targetpointpixel = self.transform_pixel2pixel(targetpointpixel)
        img = raw_img.copy()
        for traj in targetpointpixel:
            cv2.circle(img, tuple(traj), 10, (0, 255, 255), -1)
        return img
        # if is_ego:
        #     line = np.concatenate([np.zeros((1,2)),traj],axis=0)
        # else:
        #     line = traj
        # img = raw_img.copy()        
        # pts_4d = np.stack([line[:,0],line[:,1],np.zeros((line.shape[0])),np.ones((line.shape[0]))])
        # pts_2d = (self.coor2topdown @ pts_4d).T
        # pts_2d[:, 0] /= pts_2d[:, 2]
        # pts_2d[:, 1] /= pts_2d[:, 2]
        # mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
        # if not mask.any():
        #     return img
        
        # # draw raw points
        # pts_2d = pts_2d[mask,0:2]
        # for i in range(pts_2d.shape[0]):
        #     rgb_color_tuple = (255, 0, 0)
        #     if pts_2d[i,0]>0 and pts_2d[i,0]<canvas_size[1] and pts_2d[i,1]>0 and pts_2d[i,1]<canvas_size[0]:
        #         cv2.circle(img,(int(pts_2d[i,0]),int(pts_2d[i,1])),radius=4,color=rgb_color_tuple, thickness=thickness)   
        #     elif i==0:
        #         break

        # # draw line from smoothed points
        # try:
        #     tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        # except:
        #     return img
        # unew = np.linspace(0, 1, 100)
        # smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

        # num_points = len(smoothed_pts)
        # for i in range(num_points-1):
        #     hue = hue_start + (hue_end - hue_start) * (i / num_points)
        #     hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
        #     rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
        #     rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
        #     if smoothed_pts[i,0]>0 and smoothed_pts[i,0]<canvas_size[1] and smoothed_pts[i,1]>0 and smoothed_pts[i,1]<canvas_size[0]:
        #         cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)   
        #     elif i==0:
        #         break
        # return img

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
    
    def get_bounding_boxes(self):
        # ego_vehicle
        npc = self.manager.ego_vehicles[0]
        npc_id = str(npc.id)
        npc_type_id = npc.type_id
        npc_base_type = npc.attributes['base_type']
        location = npc.get_transform().location
        rotation = npc.get_transform().rotation

        extent = npc.bounding_box.extent
        center = npc.get_transform().transform(npc.bounding_box.location)

        road_id = -1
        lane_id = -1
        section_id = -1
        world2ego = npc.get_transform().get_inverse_matrix()

        result = {
            'class': 'ego_vehicle',
            'id': npc_id,
            'type_id': npc_type_id,
            'base_type': npc_base_type,
            'location': [location.x, location.y, location.z],
            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
            'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
            'center': [center.x, center.y, center.z],
            'extent': [extent.x, extent.y, extent.z],
            'semantic_tags': [npc.semantic_tags],
            'color': npc.attributes['color'],
            'road_id': road_id,
            'lane_id': lane_id,
            'section_id': section_id,
            'world2ego': world2ego,
        }
        return result

def round_two_dim_list(lst):
    return [[round(item, 3) for item in sublist] for sublist in lst]