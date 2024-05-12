add_plugin(CollectData)

target_sources(CollectData PRIVATE
    CollectData.cpp
    CollectData.h
)

target_source_group(CollectData "RenderPasses")
