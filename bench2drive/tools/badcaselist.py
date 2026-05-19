import json

# 假设你的JSON数据已加载到变量data中
with open('/home/zhanglingjun.zlj/code/Bench2Drive/mergejson/1128220_0.json', 'r') as f:
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
                shibailist.append("bench2drive220_0_1127_hybrid_"+rd['save_name'])
                break
    else:
        shibailist.append("bench2drive220_0_1127_hybrid_"+rd['save_name'])
print(shibailist)
# print(shibailist+scenario_list)
                