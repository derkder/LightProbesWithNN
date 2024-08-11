import re
import os
import json
import random

def render_graph_ReflectMapGen():
    g = RenderGraph("ReflectionMapGen")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ReflectMapGen = createPass("ReflectMapGen", {'maxBounces': 3})
    g.addPass(ReflectMapGen, "ReflectMapGen")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "ReflectMapGen.vbuffer")
    g.addEdge("VBufferRT.viewW", "ReflectMapGen.viewW")
    g.addEdge("ReflectMapGen.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("AccumulatePass.output")
    # g.markOutput("ReflectMapGen.diffuse")
    # g.markOutput("ReflectMapGen.specular")
    # g.markOutput("ReflectMapGen.roughnessemmisive")
    g.markOutput("ReflectMapGen.probePoses")
    g.markOutput("ReflectMapGen.rayDirs")
    g.markOutput("ToneMapper.dst")
    return g

def modify_translation(scene_path, json_path, line_number, x_range, y_range, z_range, idx):
    with open(scene_path, 'r') as file:
        lines = file.readlines()

    line = lines[line_number]
    if "translation=float3(" in line:
        match = re.search(r"translation=float3\(([^,]+), ([^,]+), ([^,]+)\)", line)
        if match:
            new_x = random.uniform(*x_range)
            new_y = random.uniform(*y_range)
            new_z = random.uniform(*z_range)
            lines[line_number] = re.sub(r"translation=float3\(([^,]+), ([^,]+), ([^,]+)\)",
                                        f"translation=float3({new_x:.3f}, {new_y:.3f}, {new_z:.3f})", line)
            seed_cur = random.randint(1, 4294967295)
            pos = {
                "idx": idx,
                "curSeed": seed_cur,
                "new_x": new_x,
                "new_y": new_y,
                "new_z": new_z
            }

    with open(scene_path, 'w') as file:
        file.writelines(lines)

    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = []
    
    if not isinstance(data, list):
        data = []

    # Validate pos data to ensure there are no null values
    pos = {k: (0 if v is None else v) for k, v in pos.items()}

    data.append(pos)
    
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Sphere position updated successfully")

scene_path = "C:/Files/CGProject/NNLightProbes/MyScene/cornell_box.pyscene"
output_path =  "C:/Files/CGProject/NNLightProbes/dumped_data/NormData/rawraw"
# json_path =  "C:/Files/CGProject/NNLightProbes/dumped_data/tempFullData718/raw/info.json"
# scene_path = "D:/Projects/LightProbesWithNN/MyScene/cornell_box.pyscene"
# output_path =  "D:/Projects/LightProbesWithNN/dumped_data/ShuffledData/raw"
# json_path =  "D:/Projects/LightProbesWithNN/dumped_data/tempFullData722/raw/info.json"
pos_line_idx = 47  # Adjusted to the correct line index
n_collect_frames = 100000000
n_match_frames = 3000
n_cap_offset = 10
# n_match_frames = 1000
n_sample_count = 0

ReflectMapGen = render_graph_ReflectMapGen()
try: m.addGraph(ReflectMapGen)
except NameError: pass

# modify_translation(scene_path, json_path, pos_line_idx, (-0.272, 0.272), (0.02, 0.547), (-0.272, 0.272), n_sample_count)
# m.loadScene(scene_path)

# for i in range(n_collect_frames):
#     i += 1 // woc 这里是不是一直多加了
#     renderFrame()

#     if 0 == (i % n_match_frames):
#         file_name_format = "frame_{:04d}".format(n_sample_count)
#         outputDir = f"{output_path}/{file_name_format}"
#         os.makedirs(outputDir, exist_ok=True)
#         m.frameCapture.outputDir = outputDir
#         m.frameCapture.capture()
#         n_sample_count += 1

#         modify_translation(scene_path, json_path, pos_line_idx, (-0.272, 0.272), (0.02, 0.547), (-0.272, 0.272), n_sample_count)
#         m.unloadScene()
#         m.loadScene(scene_path)



# ----------------------------------

m.loadScene(scene_path)
for i in range(n_collect_frames):
    # if(0 == i):
    #     i += 1 # 防止第一帧被保存

    renderFrame()
    if 0 == ((i + n_cap_offset) % n_match_frames):
        file_name_format = "frame_{:04d}".format(n_sample_count)
        outputDir = f"{output_path}/{file_name_format}"
        os.makedirs(outputDir, exist_ok=True)
        m.frameCapture.outputDir = outputDir
        m.frameCapture.capture()
        n_sample_count += 1

# ----------------------------------


# m.loadScene(scene_path)
# for i in range(20):
#     renderFrame()
#     file_name_format = "frame_{:04d}".format(n_sample_count)
#     outputDir = f"{output_path}/{file_name_format}"
#     os.makedirs(outputDir, exist_ok=True)
#     m.frameCapture.outputDir = outputDir
#     m.frameCapture.capture()
#     n_sample_count += 1