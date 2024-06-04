add_plugin(ReflectMapGen)

target_sources(ReflectMapGen PRIVATE
    ReflectMapGen.cpp
    ReflectMapGen.h
    ReflectMapGen.rt.slang
)

target_copy_shaders(ReflectMapGen RenderPasses/ReflectMapGen)

target_source_group(ReflectMapGen "RenderPasses")
