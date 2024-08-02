add_plugin(NeuralInputPass)

target_sources(NeuralInputPass PRIVATE
    NeuralInputPass.cpp
    NeuralInputPass.h
    NeuralInputPass.rt.slang
    ResolvePass.cs.slang
)

target_copy_shaders(NeuralInputPass RenderPasses/NeuralInputPass)

target_source_group(NeuralInputPass "RenderPasses")