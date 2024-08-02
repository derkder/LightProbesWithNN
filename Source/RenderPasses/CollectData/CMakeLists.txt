add_plugin(CollectData)

target_sources(CollectData PRIVATE
    CollectData.cpp
    CollectData.h
    Network.h
    Network.cu
    CollectData.rt.slang
    ResolvePass.cs.slang
    Config.h
    Params.slang
)

target_copy_shaders(CollectData RenderPasses/CollectData)

target_source_group(CollectData "RenderPasses")
