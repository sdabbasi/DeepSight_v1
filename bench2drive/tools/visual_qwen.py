import json
import os
import re
import cv2
import random
import numpy as np
from tqdm import tqdm


def interpolate_points(points, points_between=10):
    """
    Interpolate points between each pair of consecutive points.
    
    Args:
        points: List of 3D points as tuples (x, y, z)
        points_between: Number of points to add between each pair (default: 3)
    
    Returns:
        List of interpolated points including the original points
    """
    if len(points) < 2:
        return points
    
    interpolated = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        # Add the starting point
        if i == 0:
            interpolated.append(p1)
        
        # Calculate the step for each dimension
        dx = (p2[0] - p1[0]) / (points_between + 1)
        dy = (p2[1] - p1[1]) / (points_between + 1)
        dz = (p2[2] - p1[2]) / (points_between + 1)
        
        # Generate intermediate points
        for j in range(1, points_between + 1):
            x = p1[0] + dx * j
            y = p1[1] + dy * j
            z = p1[2] + dz * j
            interpolated.append([x, y, z])
    
    # Add the final point
    if points:
        interpolated.append(points[-1])
    
    return interpolated

def get_future_trajectory(worldpoint, current_idx, max_idx):
    trajectory = []
    
    # 尝试添加当前点和未来的几个点
    # for offset in [5, 10, 15, 20]:
    for offset in [5]:
        idx = current_idx + offset
        if str(idx) in worldpoint:  # 检查该索引是否存在
            trajectory.append(worldpoint[str(idx)]['location'])
    
    return trajectory if trajectory else None

def visual_for_crop(scene_path, visual_path, k=5):
    # 定义路径
    hz_index = list(range(0, 21, 5))
    scene_name = scene_path.split('/')[-1]
    bev_img_folders = [os.path.join(scene_path, 'camera', f'rgb_bev_{i}th-hz') for i in hz_index]
    img_files = os.listdir(bev_img_folders[0])
    img_files = [f for f in img_files if f.endswith('.jpg')]
    num_img = len(img_files)
    random_index = random.sample(list(range(num_img)), k)
    for index in random_index:
        visual_img = []
        for bev_img_folder in bev_img_folders:
            img = cv2.imread(os.path.join(bev_img_folder, f'{index:05d}.jpg'))
            visual_img.append(img)
        visual_img = np.concatenate(visual_img, axis=1)
        cv2.imwrite(os.path.join(visual_path, f'{scene_name}_{index:05d}.jpg'), visual_img)


def parse_prompt_and_answer(prompt, answer):
    # 解析 prompt
    # command = prompt.split('Mission Goal: ')[1].split(' ')[0]
    command = "4"
    his_trajs = prompt.split('Historical trajectory: ')[1].split(' ')[0]
    speed_content = prompt.split('current speed info: ')[1].split('\n')[0]
    # 解析 answer
    future_trajs_pixel = answer.split('future pixel tokens: ')[1].split('. </answer>')[0]
    future_trajs = answer.split('future waypoints: ')[1].split('. </answer>')[0]

    # 轨迹转坐标
    pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
    matches = re.findall(pattern, future_trajs)
    future_trajs = [[float(matches[i]), float(matches[i+1])] for i in range(0, len(matches), 2)]

    matches = re.findall(pattern, his_trajs)
    his_trajs = [[float(matches[i]), float(matches[i+1])] for i in range(0, len(matches), 2)]

    # 像素转坐标
    pattern = r'<\|pixel_token_([-+]?\d+)\|>'
    matches = re.findall(pattern, future_trajs_pixel)
    future_trajs_pixel = [[int(matches[i+1]), int(matches[i])] for i in range(0, len(matches), 2)]

    return command, speed_content, his_trajs, future_trajs, future_trajs_pixel


def parse_answer(answer):
    # 解析 answer
    future_trajs_pixel = answer.split('future pixel tokens: ')[1].split('. </answer>')[0]
    future_trajs = answer.split('future waypoints: ')[1].split('. </answer>')[0]

    # 轨迹转坐标
    pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
    matches = re.findall(pattern, future_trajs)
    future_trajs = [[float(matches[i]), float(matches[i+1])] for i in range(0, len(matches), 2)]

    # 像素转坐标
    pattern = r'<\|pixel_token_([-+]?\d+)\|>'
    matches = re.findall(pattern, future_trajs_pixel)
    future_trajs_pixel = [[int(matches[i+1]), int(matches[i])] for i in range(0, len(matches), 2)]

    return future_trajs, future_trajs_pixel




def transform_traj_for_visual(his_trajs, future_trajs, future_trajs_pixel):
    his_trajs = transform_traj2pixel(his_trajs)
    future_trajs = transform_traj2pixel(future_trajs)
    future_trajs_pixel = transform_pixel2pixel(future_trajs_pixel)
    return his_trajs, future_trajs, future_trajs_pixel


def transform_traj2pixel(trajs):
    coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0], 
                                    [0.0, -1.0,  0.0,  0.0], 
                                    [0.0,  0.0, -1.0, 50.0], 
                                    [0.0,  0.0,  0.0,  1.0]])
    topdown_intrinsics = np.array([[548.993771650447, 0.0, 512.0, 0], [0.0, 548.993771650447, 512.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
    coor2topdown = topdown_intrinsics @ coor2topdown

    line = np.concatenate([trajs],axis=0)
    pts_4d = np.stack([line[:,0],line[:,1],np.zeros((line.shape[0])),np.ones((line.shape[0]))])
    # print('自车坐标系轨迹：', pts_4d.tolist())
    pts_2d = (coor2topdown @ pts_4d).T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    # # 自车坐标系转相机坐标系
    # trajs = np.array(trajs) # N, 2
    # trajs = trajs[:, ::-1]
    # trajs = np.concatenate([trajs, np.zeros((trajs.shape[0], 2))], axis=1) # N, 4
    # trajs[:, 3] = 1
    # print('自车坐标系轨迹：', trajs)
    # pixel_trajs = trajs @ coor2topdown.T
    # print('相机坐标系轨迹：', pts_2d[:, :2].tolist())
    pts_2d = pts_2d[:, :2]
    pts_2d = pts_2d[:, ::-1]
    pts_2d = 1024 - pts_2d

    return pts_2d.tolist()


def transform_traj2ego(trajs, bounding_boxes):
    world2ego = np.array(bounding_boxes['world2ego'])  # 4x4
    n_traj = len(trajs)
    ones_traj = np.ones((n_traj, 1))
    trajs = np.concatenate([np.array(trajs), ones_traj], axis=1)
    # trajs = np.array(trajs + [1])  # N x 4
    trajs_ego = trajs @ world2ego.T
    return trajs_ego[:, :2].tolist()


def transform_pixel2pixel(trajs_pixel):
    # 相对坐标系转绝对坐标系
    for traj in trajs_pixel:
        traj[0] = 512 + traj[0] * 2
        traj[1] = 512 - traj[1] * 2

    return trajs_pixel


def visual_traj_on_bev(bev_img, his_trajs, future_trajs, future_trajs_pixel, route_points_pixel):
    for traj in his_trajs:
        traj = (round(traj[0]), round(traj[1]))
        cv2.circle(bev_img, tuple(traj), 5, (0, 0, 255), -1)

    for traj in future_trajs:
        traj = (round(traj[0]), round(traj[1]))
        cv2.circle(bev_img, tuple(traj), 5, (0, 255, 0), -1)

    for traj in future_trajs_pixel:
        traj = (round(traj[0]), round(traj[1]))
        cv2.circle(bev_img, tuple(traj), 2, (255, 0, 0), -1)
    
    if route_points_pixel is not None:
        for traj in route_points_pixel:
            traj = (round(traj[0]), round(traj[1]))
            cv2.circle(bev_img, tuple(traj), 10, (0, 0, 255), 2)
    
    return bev_img


def visual_for_bev(all_res, index, route_points, scene_path, visual_path=None):

    COMMAND_DICT = {
            -1: 'VOID',
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'LANEFOLLOW',
            5: 'CHANGELANELEFT',
            6: 'CHANGELANERIGHT',
        }

    # 绘制文字和可视化点：
    prompt = all_res[index]['prompt']
    answer = all_res[index]['output']
    command, speed_content, his_trajs, future_trajs, future_trajs_pixel = parse_prompt_and_answer(prompt, answer)

    # print('提取label', his_trajs, future_trajs, future_trajs_pixel)
    visual_text = f'Mission Goal: {command} {COMMAND_DICT[int(command)]}\ncurrent speed info: {speed_content}\nHistorical trajectory: {his_trajs}' + \
                  f'Future trajectory pixel: {future_trajs_pixel}\nFuture trajectory: {future_trajs}'
    bev_img = os.path.join(scene_path, 'camera', 'CAM_BEV', f'{index:05}.jpg')
    his_trajs, future_trajs, future_trajs_pixel = transform_traj_for_visual(his_trajs, future_trajs, future_trajs_pixel)
    if route_points is not None:
        # route_points = interpolate_points(route_points)
        # print('world', route_points)
        # route_points = interpolate_points(route_points)
        route_points = transform_traj2ego(route_points, all_res[index]["bounding_boxes"])
        # print('ego', route_points)
        route_points_pixel = transform_traj2pixel(route_points)
        # print('pixel', route_points_pixel)
        # inputs = input('press any key to continue')
    else:
        route_points_pixel = None

    # print('像素坐标', his_trajs, future_trajs, future_trajs_pixel)
    bev_img = cv2.imread(bev_img)
    # print(bev_img.shape)
    bev_img = visual_traj_on_bev(bev_img, his_trajs, future_trajs, future_trajs_pixel, route_points_pixel)

    front_img = os.path.join(scene_path, 'camera', 'CAM_FRONT', f'{index:05}.jpg')
    front_img = cv2.imread(front_img)
    h, w = front_img.shape[:2]
    dst_h = bev_img.shape[0]
    dst_w = int(w * dst_h / h)
    front_img = cv2.resize(front_img, (dst_w, dst_h))
    visual_img = np.concatenate([front_img, bev_img], axis=1)
    # 绘制文字和可视化BEV图
    cv2.putText(visual_img, visual_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if visual_path is not None:
        cv2.imwrite(os.path.join(visual_path, f'{index:05}.jpg'), visual_img)


def main_for_eval_bench2drive():
    with open('/home/zhanglingjun.zlj/code/Bench2Drive/mergejson1225/merged.json', 'r') as f:
        data = json.load(f)
    scenario_list = []
    # 检查JSON结构并提取所需字段
    exceptions = data.get("_checkpoint", {}).get("global_record", {}).get("meta", {}).get("exceptions", [])
    shibailist = []
    records = data['_checkpoint']['records']
    for rd in records:
        if rd['status']=='Completed' or rd['status']=='Perfect':
            success_flag = True
            for k,v in rd['infractions'].items():
                if len(v)>0 and k != 'min_speed_infractions':
                    success_flag = False
                    shibailist.append(rd['save_name'])
                    break
        else:
            shibailist.append(rd['save_name'])
    # print(shibailist)
    results_path = '/home/zhanglingjun.zlj/code/Bench2Drive/result6/Scenarios'
    video_path = '/home/zhanglingjun.zlj/code/Bench2Drive/videores1'
    os.makedirs(video_path, exist_ok=True)
    # scene_paths = os.listdir(results_path)
    scene_paths = shibailist
    scene_paths = [os.path.join(results_path, scene_path) for scene_path in scene_paths if os.path.isdir(os.path.join(results_path, scene_path))]
    # print(scene_paths)
    # route_points = json.load(open('routes.json'))[0]["waypoints"]
    # route_points = [[float(point['x']), float(point['y']), float(point['z'])] for point in route_points]
    route_points = None
    for scene_path in scene_paths:
        scene_key = scene_path.split('/')[-1]
        # if 'BlockedIntersection' not in scene_path:
        #     continue
        print(scene_path)
        all_mid_results = os.listdir(scene_path + '/meta')
        all_mid_results = [os.path.join(scene_path + '/meta', mid_result) for mid_result in all_mid_results if mid_result.endswith('.json')]
        all_mid_res = [json.load(open(mid_result, 'r')) for mid_result in all_mid_results]
        location_json_path = os.path.join(scene_path, 'metric_info.json')
        with open(location_json_path, 'r') as file:
            worldpoint = json.load(file)
        # bev 视角可视化
        num_frame = len(all_mid_res)
        visual_path = os.path.join(scene_path, 'visual_bev')
        os.makedirs(visual_path, exist_ok=True)
        for i in tqdm(range(num_frame - 1)):
            max_idx = max(int(k) for k in worldpoint.keys())  # 获取最大的索引
            route_points = get_future_trajectory(worldpoint, i, max_idx)
            visual_for_bev(all_mid_res, i, route_points, scene_path, visual_path)

        # 将可视化的图转为视频
        visual_img_paths = os.listdir(visual_path)
        visual_img_paths = [os.path.join(visual_path, visual_img_path) for visual_img_path in visual_img_paths if visual_img_path.endswith('.jpg')]
        visual_img_paths = sorted(visual_img_paths)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_file = os.path.join(video_path, f'{scene_key}.mp4')
        video = cv2.VideoWriter(video_file, fourcc, 20, (1280, 512))
        for visual_img_path in tqdm(visual_img_paths):
            visual_img = cv2.imread(visual_img_path)
            visual_img = cv2.resize(visual_img, (1280, 512))
            video.write(visual_img)
        video.release()


if __name__ == '__main__':
    # main_for_vis_infer()
    # main_for_vis_train()
    # main_for_eval_l2()
    main_for_eval_bench2drive()