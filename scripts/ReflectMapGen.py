from falcor import *
import re
import os
import json
import random

def render_graph_ReflectMapGen():
    g = RenderGraph("ReflectionMapGen")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    # ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    # g.addPass(ToneMapper, "ToneMapper")
    ReflectMapGen = createPass("ReflectMapGen", {'maxBounces': 3})
    g.addPass(ReflectMapGen, "ReflectMapGen")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    # g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "ReflectMapGen.vbuffer")
    g.addEdge("VBufferRT.viewW", "ReflectMapGen.viewW")
    g.addEdge("ReflectMapGen.color", "AccumulatePass.input")
    g.markOutput("AccumulatePass.output")
    return g

# we could add random material param later
def modify_translation(scene_path, json_path, line_number, x_range, y_range, z_range):
    # change pos in pyscene
    with open(scene_path, 'r') as file:
        lines = file.readlines()

    # find line to sphere probe pos
    line = lines[line_number]
    if "translation=float3(" in line:
        match = re.search(r"translation=float3\(([^,]+), ([^,]+), ([^,]+)\)", line)
        if match:
            # generate ner pos
            new_x = random.uniform(*x_range)
            new_y = random.uniform(*y_range)
            new_z = random.uniform(*z_range)
            lines[line_number] = re.sub(r"translation=float3\(([^,]+), ([^,]+), ([^,]+)\)",
                                        f"translation=float3({new_x:.3f}, {new_y:.3f}, {new_z:.3f})", line)
            pos = {
                "new_x": new_x,
                "new_y": new_y,
                "new_z": new_z
            }

    with open(scene_path, 'w') as file:
        file.writelines(lines)

    # record our new sphere pos in json file
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = []
    data.append(pos)
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print("sphere pos updated successfully")


# scene_path = "C:/Files/CGProject/NNLightProbes/MyScene/cornell_box.pyscene"
scene_path = "D:/Projects/LightProbesWithNN/MyScene/cornell_box.pyscene"
# output_path =  "C:/Files/CGProject/NNLightProbes/dumped_data"
output_path = "D:/Projects/LightProbesWithNN/dumped_data/temp"
# json_path =  "C:/Files/CGProject/NNLightProbes/dumped_data/info.json"
json_path = "D:/Projects/LightProbesWithNN/dumped_data/temp/info.json"
pos_line_idx = 48 - 1 # file read idx start from 0 while vs_window start from 0 
n_collect_frames = 30000000
n_match_frames = 100000
n_sample_count = 7

ReflectMapGen = render_graph_ReflectMapGen()
try: m.addGraph(ReflectMapGen)
except NameError: None

modify_translation(scene_path, json_path, pos_line_idx, (-0.272, 0.272), (0.02, 0.547), (-0.272, 0.272))
m.loadScene(scene_path)

for i in range(n_collect_frames):
    i += 1 # 防止渲染的第一帧就要被保存下来
    renderFrame()
    if 0 == (i % n_match_frames):
        # save output
        file_name_format = "frame_{:04d}".format(n_sample_count)
        outputDir = f"{output_path}/{file_name_format}"
        os.makedirs(outputDir, exist_ok=True)
        m.frameCapture.outputDir = outputDir
        m.frameCapture.capture()
        n_sample_count += 1

        # move the probe and reload scene
        modify_translation(scene_path, json_path, 47, (-0.272, 0.272), (0.02, 0.547), (-0.272, 0.272))
        m.unloadScene()
        m.loadScene(scene_path)


