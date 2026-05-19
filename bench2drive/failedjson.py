import json
import xml.etree.ElementTree as ET
import re
import os

def get_failed_route_ids(json_path):
    """
    根据提供的逻辑从 JSON 中提取失败的 route_id (数字部分)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    shibai_ids = set()
    records = data.get('_checkpoint', {}).get('records', [])
    
    for rd in records:
        route_id_full = rd['route_id']  # 格式如 "RouteScenario_4937_rep0"
        
        # 提取中间的数字 ID (例如 4937)
        match = re.search(r'RouteScenario_(\d+)_rep', route_id_full)
        if not match:
            continue
        pure_id = match.group(1)
        
        is_success = False
        if rd['status'] in ['Completed', 'Perfect']:
            is_success = True
            for k, v in rd['infractions'].items():
                # 如果有违章（排除低速违章），则视为失败
                if len(v) > 0 and k != 'min_speed_infractions':
                    is_success = False
                    break
        else:
            is_success = False
            
        if not is_success:
            shibai_ids.add(pure_id)
            
    print(f"找到失败/未完成的路线数量: {len(shibai_ids)}")
    return shibai_ids

def filter_xml_by_ids(input_xml, output_xml, target_ids):
    """
    从原始 XML 中提取匹配 target_ids 的 route 节点
    """
    if not os.path.exists(input_xml):
        print(f"错误: 找不到输入 XML 文件 {input_xml}")
        return

    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # 创建一个新的根节点
    new_root = ET.Element('routes')
    
    count = 0
    # 遍历 XML 中所有的 route
    for route in root.findall('route'):
        route_id = route.get('id')
        if route_id in target_ids:
            new_root.append(route)
            count += 1
    
    # 保存新 XML
    new_tree = ET.ElementTree(new_root)
    
    # 这里的 indent 主要是为了美化输出 (Python 3.9+)
    if hasattr(ET, 'indent'):
        ET.indent(new_tree, space="   ", level=0)
        
    new_tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print(f"成功提取 {count} 条路线到 {output_xml}")

if __name__ == "__main__":
    # 配置路径
    JSON_FILE = '/home/zhanglingjun.zlj/code/Bench2Drive/cotmerge/merged.json'
    INPUT_XML = '/home/zhanglingjun.zlj/code/Bench2Drive/leaderboard/data/bench2drive220.xml' # 请修改为你原始的 XML 路径
    OUTPUT_XML = '/home/zhanglingjun.zlj/code/Bench2Drive/leaderboard/data/failed2.xml'
    
    # 1. 获取失败的 ID 集合
    failed_ids = get_failed_route_ids(JSON_FILE)
    
    # 2. 过滤 XML
    filter_xml_by_ids(INPUT_XML, OUTPUT_XML, failed_ids)
