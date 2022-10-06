import os

tasks=[1,2]

for task in tasks:
    script=f"CUDA_VISIBLE_DEVICES=1 nohup python train_graph_property.py --task={str(task)} > gp{task} 2>&1 "
    os.system(script)
    script=f"CUDA_VISIBLE_DEVICES=1 nohup python train_node_property.py --task={str(task)} > np{task} 2>&1 "
    os.system(script)

