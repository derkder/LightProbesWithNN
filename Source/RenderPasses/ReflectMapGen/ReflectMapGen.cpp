/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ReflectMapGen.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <ctime>

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReflectMapGen>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/ReflectMapGen/ReflectMapGen.rt.slang";
const char kResolveFile[] = "RenderPasses/ReflectMapGen/ResolvePass.cs.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
//const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxPayloadSizeBytes = 160u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";
const char kOutputColor[] = "color";
const char kOutputProbePoses[] = "probePoses";
const char kOutputRayDirs[] = "rayDirs";
const char kOutputHitNormals[] = "hitNormals";
const char kOutputHitMaterials[] = "hitMaterials";

const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" },
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { kOutputColor,          "gOutputColor",      "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    { kOutputProbePoses,      "gOutputProbePoses",  "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    { kOutputRayDirs,      "gOutputRayDirs",  "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    { kOutputHitNormals,      "gOutputHitNormals",  "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    { kOutputHitMaterials,      "gOutputHitMaterials",  "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float }
    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";
const char kIsCutting[] = "isCutting";//normal rayTrace or lightprobe raytrace
const char kIsPerspective[] = "isPerspective"; // Perspective and orthography
} // namespace

ReflectMapGen::ReflectMapGen(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void ReflectMapGen::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else if (key == kIsCutting)
            mIsCutting = value;
        else
            logWarning("Unknown property '{}' in ReflectMapGen properties.", key);
    }
}

Properties ReflectMapGen::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kIsCutting] = mIsCutting;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

RenderPassReflection ReflectMapGen::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void ReflectMapGen::updateValue()
{
    auto currentTime = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = currentTime - lastUpdateTime;

    if (elapsed.count() >= 3.f)
    {                      
        sliceZPercent += 0.02f; 
        if (sliceZPercent > 0.8f)
        {
            sliceZPercent = 0.5f; 
        }
        lastUpdateTime = currentTime; 
    }

    std::cout << sliceZPercent;
}

void ReflectMapGen::prepareResolve(const RenderData& renderData)
{
    auto var = mpResolvePass->getRootVar();
    var["radiance"] = mOutputTex; // 新建的Texture，想把结果Blit到这里
    var["output"] = renderData.getTexture(kOutputColor);// 这个pass计算的结果
}

int generateRandomNumber(int min, int max)
{
    static unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    std::default_random_engine generator(seed++);
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}


void ReflectMapGen::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (0 == mFrameCount % frameCap)
    {
        //avoid overflow in slang
        mSeed1 = generateRandomNumber(0, 210590); 
        std::cout << "cur seed: " << mSeed1 << std::endl;
        mSeed2 = generateRandomNumber(0, 210590); 
        std::cout << "cur seed: " << mSeed2 << std::endl;
        mSeed3 = generateRandomNumber(0, 210590); 
        std::cout << "cur seed: " << mSeed3 << std::endl;
    }


    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    const auto& pInputViewDir = renderData.getTexture(kInputViewDir);
    if (!mOutputTex)
    {
        mOutputTex = mpDevice->createTexture2D(
            pInputViewDir->getWidth(),
            pInputViewDir->getHeight(),
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource
        );
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("IS_CUTTING", mIsCutting ? "1" : "0");
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gSceneAABBCenter"] = mSceneAABBCenter;
    var["CB"]["gSceneAABBExtent"] = mSceneAABBExtent;
    var["CB"]["gIntervalX"] = mIntervalX;
    var["CB"]["gIntervalY"] = mIntervalY;
    var["CB"]["gCurZ"] = minZ + (maxZ - minZ) * sliceZPercent; // posZ of light probe
    var["CB"]["gIsCutting"] = mIsCutting;
    var["CB"]["gSeed1"] = mSeed1;
    var["CB"]["gSeed2"] = mSeed2;
    var["CB"]["gSeed3"] = mSeed3;
    var["CB"]["kProbeLoc"] = mProbeLoc;

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels)
        bind(channel);
    for (auto channel : kOutputChannels)
        bind(channel);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    const uint2 rayDispatchDim = uint2(sampleNumsX, sampleNumsY);
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Spawn the rays.
    // 这里应该改一下，不是生成像素点那么多的光线了
    // 在目前这里最后一个参数还真的还是1，但是到后面就不是了噢
    // 这里targetdim想要投射到空间中对应的点上
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(rayDispatchDim, 1));

    //mpResolvePass->execute(pRenderContext, Falcor::uint3(targetDim, 1));
    //auto ext = Bitmap::getFileExtFromResourceFormat(mOutputTex->getFormat());
    //auto fileformat = Bitmap::getFormatFromFileExtension(ext);
    //Bitmap::ExportFlags flags = Bitmap::ExportFlags::None;
    //flags |= Bitmap::ExportFlags::ExportAlpha;
    //std::string path = "D:/Projects/temp/wowwowowowow.ext";
    //mOutputTex->captureToFile(0, 0, path, fileformat, flags, false /* async */);

    mFrameCount++;
    //updateValue();
}

void ReflectMapGen::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Is Cutting", mIsCutting);
    widget.tooltip("Is Cutting", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void ReflectMapGen::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    mpResolvePass = nullptr;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ReflectMapGen: This render pass does not support custom primitives.");
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        //desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxPayloadSize(160);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid),
                desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());

        ProgramDesc resolveDesc;
        resolveDesc.addShaderLibrary(kResolveFile).csEntry("main");
        resolveDesc.addShaderModules(mpScene->getShaderModules());
        resolveDesc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpSampleGenerator->getDefines());
        defines.add(mpScene->getSceneDefines());
        mpResolvePass = ComputePass::create(mpDevice, resolveDesc, defines);

         //std::ifstream file(json_path);
         //if (!file.is_open())
         //{
         //    std::cout << "Failed to open JSON file" << std::endl;
         //}
         //nlohmann::json data;
         //file >> data;
         //if (data.empty())
         //{
         //    std::cout << "JSON file is empty" << std::endl;
         //}

         //// 获取最后一个元素
         //auto latest = data.back();
         //mProbeLoc = float3(latest["new_x"], latest["new_y"], latest["new_z"]);
         //mSeed = latest["curSeed"];
         //std::cout << "mProbeLoc: " << mProbeLoc.x << "  " << mProbeLoc.y << "  " << mProbeLoc.z << std::endl;
         //mProbeLoc = float3(-0.056081911802012024, 0.1454754890310627, 0.24149841116601556);
         //mSeed = 1406584362;
        //mSeed = static_cast<unsigned int>(std::time(0));
        //-0.056081911802012024, "new_y" : 0.1454754890310627, "new_z" : 0.24149841116601556
        mSceneAABBCenter = mpScene->getSceneBounds().center();
        mSceneAABBExtent = mpScene->getSceneBounds().extent();
        std::cout << "mSceneAABBCenter: " << mSceneAABBCenter.x << "  " << mSceneAABBCenter.y << "  " << mSceneAABBCenter.z << std::endl;
        std::cout << "mSceneAABBExtent: " << mSceneAABBExtent.x << "  " << mSceneAABBExtent.y << "  " << mSceneAABBExtent.z << std::endl;
    }
}

void ReflectMapGen::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}


