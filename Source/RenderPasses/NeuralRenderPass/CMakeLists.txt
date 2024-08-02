add_plugin(NeuralRenderPass)

target_sources(NeuralRenderPass PRIVATE
    NeuralRenderPass.cpp
    NeuralRenderPass.h
    NeuralRenderPass.rt.slang
    ResolvePass.cs.slang
)

target_copy_shaders(NeuralRenderPass RenderPasses/NeuralRenderPass)

target_source_group(NeuralRenderPass "RenderPasses")