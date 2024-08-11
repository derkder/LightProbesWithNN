add_plugin(NetworkPass)

target_sources(NetworkPass PRIVATE
    NetworkPass.cpp
    NetworkPass.h
    Network.rt.slang
)

target_copy_shaders(NetworkPass RenderPasses/NetworkPass)
target_source_group(NetworkPass "RenderPasses")
