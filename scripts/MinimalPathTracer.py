from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("MinimalPathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    MinimalPathTracer = createPass("MinimalPathTracer", {'maxBounces': 3})
    g.addPass(MinimalPathTracer, "MinimalPathTracer")

    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")

    g.markOutput("MinimalPathTracer.hitposes")
    g.markOutput("MinimalPathTracer.raydirs")
    g.markOutput("ToneMapper.dst")
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None

n_collect_frames = 10000
n_match_frames = 3000
n_cap_offset = 1
# n_match_frames = 1000
n_sample_count = 0
scene_path = "C:/Projects/Graphics/FalcorTCNN/MyScene/cornell_box.pyscene"
output_path =  "C:/Files/CGProject/NNLightProbes/dumped_data/TestData/raw"

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