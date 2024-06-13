from falcor import *

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

ReflectMapGen = render_graph_ReflectMapGen()
try: m.addGraph(ReflectMapGen)
except NameError: None
