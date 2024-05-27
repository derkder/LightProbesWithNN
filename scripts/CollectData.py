from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("MinimalPathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    MinimalPathTracer = createPass("MinimalPathTracer", {'maxBounces': 3})
    g.addPass(MinimalPathTracer, "MinimalPathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")

    # g.markOutput("MinimalPathTracer.color")
    g.markOutput("ToneMapper.dst")
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None

# m.loadScene("C:/Files/CGProject/NNLightProbes/media/inv_rendering_scenes/bunny_init.pyscene")


n_collect_frames = 5000

for i in range(n_collect_frames):
    if(0 == i % 100):
        outputDir = "C:/Files/CGProject/MyTemp/frame_{:04d}".format(i)
        os.makedirs(outputDir, exist_ok=True)
        m.frameCapture.outputDir = outputDir
        renderFrame()
        m.frameCapture.capture()


# from falcor import *

# def render_graph_MinimalPathTracer():
#     g = RenderGraph("MinimalPathTracer")
#     # AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
#     # g.addPass(AccumulatePass, "AccumulatePass")
#     # ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
#     # g.addPass(ToneMapper, "ToneMapper")
#     MinimalPathTracer = createPass("MinimalPathTracer", {'maxBounces': 3})
#     g.addPass(MinimalPathTracer, "MinimalPathTracer")
#     VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
#     g.addPass(VBufferRT, "VBufferRT")
#     # g.addEdge("AccumulatePass.output", "ToneMapper.src")
#     g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
#     g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
#     # g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")

#     g.markOutput("MinimalPathTracer.color")
#     # g.markOutput("ToneMapper.dst")
#     return g

# MinimalPathTracer = render_graph_MinimalPathTracer()
# try: m.addGraph(MinimalPathTracer)
# except NameError: None

# m.loadScene("C:/Files/CGProject/NNLightProbes/media/inv_rendering_scenes/bunny_init.pyscene")


# n_collect_frames = 5000

# for i in range(n_collect_frames):
#     if(0 == i % 100):
#         outputDir = "C:/Files/CGProject/MyTemp/frame_{:04d}".format(i)
#         os.makedirs(outputDir, exist_ok=True)
#         m.frameCapture.outputDir = outputDir
#         renderFrame()
#         m.frameCapture.capture()