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

/**
 * Minimal path tracer.
 *
 * The purpose is to use it for validation of more complex renderers.
 * The implementation here should be kept as simple/naive as possible.
 *
 * At each hit point (including the primary hit loaded from the V-buffer),
 * analytic light sources (point, directional) are sampled uniformly using
 * 1 shadow ray, and 1 scatter ray is traced to sample the hemisphere.
 * At hit/miss the scatter ray includes light from emissive surface and
 * the environment map, respectively. Traversal stops at a fixed path length.
 *
 * Each type of light (analytic, emissive, env map) can be individually
 * enabled/disabled from the host. This clutters the code a bit, but it is
 * important as not all other renderes may support all three light types.
 *
 * The host sets the following defines:
 *
 * MAX_BOUNCES             Maximum number of indirect bounces (0 means no indirect).
 * COMPUTE_DIRECT          Nonzero if direct illumination should be included.
 * USE_IMPORTANCE_SAMPLING Nonzero if importance sampling should be used for sampling materials.
 * USE_ANALYTIC_LIGHTS     Nonzero if Falcor's analytic lights should be used.
 * USE_EMISSIVE_LIGHTS     Nonzero if emissive geometry should be used as lights.
 * USE_ENV_LIGHT           Nonzero if env map is available and should be used as light source.
 * USE_ENV_BACKGROUND      Nonzero if env map is available and should be used as background.
 * is_valid_<name>         1 if optional I/O buffer with this name should be used.
 */

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    uint gSeed;

    float3 gSceneAABBCenter;
    float3 gSceneAABBExtent;
    float3 kProbeLoc;
    float gIntervalX;
    float gIntervalY;
    float gCurZ;

    bool gIsCutting;

}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

// Outputs
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputDiffuse;
RWTexture2D<float4> gOutputSpecular;
RWTexture2D<float4> gOutputRoughEmmi;
RWTexture2D<float4> gOutputProbePoses;
RWTexture2D<float4> gOutputRayDirs;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
#define From_Left 0
#define From_Right 1
#define From_Back 2

static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = true;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

//反过来不行是因为有球挡住了
static const bool kIsInverted = false;
static const int currentState = From_Left;
static const int frameCap = 3000;
static const int offsetCap = 1;

// 线性同余生成器参数
static const int A = 1664525;
static const int C = 1013904223;
static const int M = 2147483647; // 使用较大的质数作为模数

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

/**
 * Payload for scatter ray (up to 72B).
 */
struct ScatterRayData
{
    float3 diffuse;
    float isEmissive;
    float3 specular;
    float roughness;
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    bool terminated;  ///< Set to true when path is terminated.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    uint pathLength;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    //float reverseEmmisiveFact;
    float3 direction; ///< Next path segment direction.
    HitInfo hitInfo;

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.diffuse = float3(0, 0, 0);
        this.isEmissive = false;
        this.specular = float3(0, 0, 0);
        this.roughness = 0;
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.sg = sg;
        this.hitInfo = {};
    }
};

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @param[in] lod Method for computing texture level-of-detail.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    return sd;
}

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera)
{
    if (is_valid(gViewW))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gViewW[launchIndex].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 */
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

/**
 * Evaluates the direct illumination from analytic lights.
 * This function samples Falcor's light list uniformly with one shadow ray.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in,out] sg SampleGenerator object.
 * @return Outgoing radiance in view direction.
 */
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return float3(0.f);

    // Pick one of the analytic light sources randomly with equal probability.
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    // Sample local light source.
    AnalyticLightSample ls;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls))
        return float3(0.f);

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(origin, ls.dir, ls.distance);
    if (!V)
        return float3(0.f);

    // Evaluate contribution.
    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
}

/**
 * Generate a new scatter ray or terminate.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] isCurveHit True if on curve hit.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, bool isCurveHit, float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.thp *= bsdfSample.weight;
        return any(rayData.thp > 0.f);
    }

    return false;
}

/**
 * Process a hit.
 * Loads the shading data, samples analytic lights and samples a new scatter ray.
 * Terminates the path if maximum number of bounces is reached.
 * @param[in] hit Hit info.
 * @param[in,out] rayData Ray payload.
 *
 */
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    const bool isCurveHit = hit.getType() == HitType::Curve;
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction, lod);

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    if(1 == rayData.pathLength)
    {
        rayData.diffuse = mi.getProperties(sd).diffuseReflectionAlbedo;
        rayData.specular = mi.getProperties(sd).specularReflectionAlbedo;
        rayData.roughness = mi.getProperties(sd).roughness;
        if(length(mi.getProperties(sd).emission) > 0.01)
        {
            rayData.isEmissive = 1.0;
            //rayData.reverseEmmisiveFact = 1 / mi.getProperties(sd).emission;
        }
        else
        {
            rayData.isEmissive = 0.0;
        }
    }

    // Add emitted light.
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    {
        rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
    }

    // Check whether to terminate based on max depth.
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;
    if (isCurveHit)
    {
        // For curves, we set the new origin at the sphere center.
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else
    {
        rayOrigin = sd.computeRayOrigin();
    }

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
    {
        rayData.terminated = true;
        return;
    }

    rayData.pathLength++;
}

void myHandleHit(ShadingData sd, inout ScatterRayData rayData)
{
    let lod = ExplicitLodTextureSampler(0.f);
    bool isCurveHit = false;

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    if(1 == rayData.pathLength)
    {
        rayData.diffuse = mi.getProperties(sd).diffuseReflectionAlbedo;
        rayData.specular = mi.getProperties(sd).specularReflectionAlbedo;
        rayData.roughness = mi.getProperties(sd).roughness;
        if(length(mi.getProperties(sd).emission) > 0.01)
        {
            rayData.isEmissive = 1.0;
            //rayData.reverseEmmisiveFact = 1 / mi.getProperties(sd).emission;
        }
        else
        {
            rayData.isEmissive = 0.0;
        }
    }

    // Add emitted light.
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    {
        rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
    }

    // Check whether to terminate based on max depth.
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;
    if (isCurveHit)
    {
        // For curves, we set the new origin at the sphere center.
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else
    {
        rayOrigin = sd.computeRayOrigin();
    }

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
    {
        rayData.terminated = true;
        return;
    }

    rayData.pathLength++;
}

float3 MyTracePathMap(const uint2 pixel, const uint2 frameDim)
{
    //return float3(1.0f);
    // float3 mProbeLoc = float3(0.16089, 0.28183, 0.08310);
    //float3 mProbeLoc = float3(-0.085158, 0.4904897, -0.12996);
    float3 mProbeLoc = float3(-0.240646, 0.020569, 0.082068);
    float radius = 0.005f; // 0.005(sphere radius) + 0.002(bias)
    float3 outColor = float3(0.f);
    float theta = M_PI * pixel.y / frameDim.y;
    float phi = 2 * M_PI * pixel.x / frameDim.x;
    float x = radius * sin(theta) * cos(phi);
    float z = radius * sin(theta) * sin(phi);
    float y = radius * cos(theta);
    float3 primaryRayOrigin = float3(x, y, z) + mProbeLoc;
    const float3 primaryRayDir = - normalize(float3(x, y, z));

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount); //random number generator
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;
    float3 radiance0 = float3(0.f);
    float3 radiance1 = float3(0.f);
    for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
    {
        // if (1 == depth) return rayData.radiance;
        if (0 == depth)
        {
            VertexData v = {};
            v.posW = mProbeLoc;
            //v.normalW = -primaryRayDir;
            v.normalW = normalize(float3(1., 2., 3.));
            v.faceNormalW = v.normalW;
            v.texC = float2(0.1, 0.1);
            //v.tangent
            float3 arbitraryVector = (abs(v.normalW.y) < 0.999f) ? float3(0, 1, 0) : float3(1, 0, 0);
            float3 tangent = normalize(cross(arbitraryVector, v.normalW));
            float3 bitangent = cross(v.normalW, tangent);
            float handedness = (dot(cross(v.normalW, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
            v.tangentW = float4(tangent, handedness);
            v.curveRadius = 0;
            v.coneTexLODValue = 0;
            let lod = ExplicitLodTextureSampler(0.f);
            uint matID = 0;
            ShadingData sdTemp = gScene.materials.prepareShadingData(v, matID/*materialID*/, -primaryRayDir, lod/*lod*/);
            myHandleHit(sdTemp, rayData);
        }
        else
        {
            traceScatterRay(rayData);
            if (1 == depth)
            {
                radiance1 = rayData.radiance - radiance0;   
                gOutputDiffuse[pixel].xyz = rayData.diffuse;
                gOutputSpecular[pixel].xyz = rayData.specular;
                gOutputRoughEmmi[pixel].x = rayData.roughness;
                gOutputRoughEmmi[pixel].y = rayData.isEmissive;
            } 
        }
        
    }

    outColor += rayData.radiance; // accumulate
    return outColor;
}

float3 MyTracePathRand(const uint2 pixel, const uint2 frameDim, float3 dir, float3 probePos)
{
    //return float3(1.0f);
    //float3 mProbeLoc = float3(0.26089, 0.48183, 0.08310);
    //float3 mProbeLoc = float3(0.16089, 0.28183, 0.08310);
    float radiusH = 0.005f; // 0.005(sphere radius) + 0.002(bias)
    float radiusO = 0.006f; // 0.005(sphere radius) + 0.002(bias)
    float3 outColor = float3(0.f);
    // float theta = M_PI * pixel.y / frameDim.y;
    // float phi = 2 * M_PI * pixel.x / frameDim.x;
    // float x = radius * sin(theta) * cos(phi);
    // float z = radius * sin(theta) * sin(phi);
    // float y = radius * cos(theta);
    float3 primaryRayOrigin = probePos + dir * radiusO;
    const float3 primaryRayDir = -dir;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount); //random number generator
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = probePos + dir * radiusH;
    rayData.direction = primaryRayDir;
    float3 radiance0 = float3(0.f);
    float3 radiance1 = float3(0.f);
    for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
    {
        // if (1 == depth) return rayData.radiance;
        if (0 == depth)
        {
            VertexData v = {};
            v.posW = probePos; // 不能干脆用probePos的原因是tangent可能会错？也不一定错
            v.normalW = -primaryRayDir;
            v.faceNormalW = v.normalW;
            v.texC = float2(0.1, 0.1);
            //v.tangent
            float3 arbitraryVector = (abs(v.normalW.y) < 0.999f) ? float3(0, 1, 0) : float3(1, 0, 0);
            float3 tangent = normalize(cross(arbitraryVector, v.normalW));
            float3 bitangent = cross(v.normalW, tangent);
            float handedness = (dot(cross(v.normalW, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
            v.tangentW = float4(tangent, handedness);
            v.curveRadius = 0;
            v.coneTexLODValue = 0;
            let lod = ExplicitLodTextureSampler(0.f);
            uint matID = 0;
            ShadingData sdTemp = gScene.materials.prepareShadingData(v, matID/*materialID*/, -primaryRayDir, lod/*lod*/);
            myHandleHit(sdTemp, rayData);
        }
        else
        {
            traceScatterRay(rayData);
            if (1 == depth)
            {
                radiance1 = rayData.radiance - radiance0;   
                gOutputDiffuse[pixel].xyz = rayData.diffuse;
                gOutputSpecular[pixel].xyz = rayData.specular;
                gOutputRoughEmmi[pixel].x = rayData.roughness;
                gOutputRoughEmmi[pixel].y = rayData.isEmissive;
            } 
        }
        
    }

    outColor += rayData.radiance; // accumulate
    return outColor;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0))
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += rayData.thp * Le;
    }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    // if(0 ==  materialID && 0 != rayData.pathLength)
    //     IgnoreHit();
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    //这里的triangleHit.barycentrics和primitiveIndex，就是为了得到一个vertexData
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout ScatterRayData rayData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(curveHit), rayData);
}

//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout ScatterRayData rayData, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    handleHit(HitInfo(sdfGridHit), rayData);
}

//
// Shader entry point for ray generation shader.
//

bool shouldReturnTrue(int x, int y, int seed, uint2 frameD) {
    int totalCalls = frameD.x * frameD.y;
    int maxTrueCount = 250000;
    int h = x * 374761393 + y * 668265263 + seed * 1013904223; // Arbitrary constants
    h = (h ^ (h >> 13)) * 1274126177;
    h =  h ^ (h >> 16);
    int randomValue = abs(h) % totalCalls;

    // If random value is less than maxTrueCount, return true
    return randomValue < maxTrueCount;
}

uint hash(uint x, uint y, uint frame) {
    uint h = x + y * 40503u + frame * 20183u; // 使用较小的素数
    h ^= h >> 13;
    h *= 18043u; // 较小的素数
    h ^= h >> 8;
    h *= 24103u; // 较小的素数
    h ^= h >> 9;
    return h;
}

// 生成一个随机的归一化方向向量
float3 generateRandomNormalizedDirection(uint x, uint y, uint frame)
{
    // 使用坐标 (x, y) 和帧数 frame 生成种子
    uint seed = hash(x, y, frame);

    // 使用线性同余生成器 (LCG) 生成伪随机数
    seed = (1664525 * seed + 1013904223);
    float xRand = (float(seed & 0xFFFFFF) / float(0xFFFFFF)) * 2.0 - 1.0;

    seed = (1664525 * seed + 1013904223);
    float yRand = (float(seed & 0xFFFFFF) / float(0xFFFFFF)) * 2.0 - 1.0;

    seed = (1664525 * seed + 1013904223);
    float zRand = (float(seed & 0xFFFFFF) / float(0xFFFFFF)) * 2.0 - 1.0;

    // 归一化向量
    float length = sqrt(xRand * xRand + yRand * yRand + zRand * zRand);
    return float3(xRand / length, yRand / length, zRand / length);
}

// Xorshift 伪随机数生成器 让下面的随机数生成和上面的区分开来
uint xorshift32(uint state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// 生成三个不同的从-0.5到0.5之间的数字
float3 generateThreeUniqueRandomNumbers(uint x, uint y, uint frame)
{
    // 使用坐标 (x, y) 和帧数 frame 生成种子
    uint seed = hash(x, y, frame);

    float3 result;

    // 使用 Xorshift 生成伪随机数
    seed = xorshift32(seed);
    result.x = (float(seed & 0xFFFFFF) / float(0xFFFFFF)) - 0.5;

    seed = xorshift32(seed);
    result.y = (float(seed & 0xFFFFFF) / float(0xFFFFFF)) - 0.5;

    seed = xorshift32(seed);
    result.z = (float(seed & 0xFFFFFF) / float(0xFFFFFF)) - 0.5;

    return result;
}

//下面的东西不会不是一帧跑完所有的再接着跑下一帧的吧
//但有一说一我感觉应该不是
//可以测试一下，返回1 / gFrameCount的颜色看看是不是一直一样的
//即使不是也可以用种子控制读写
[shader("raygeneration")] 
void rayGen()
{
    // uint2 pixel = DispatchRaysIndex().xy;
    // gOutputProbePoses[pixel] = float4((float)gFrameCount / 10);

    //----------------

    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy; // frame的大小,这里改成lightProbes的数量

    float3 dir;
    float3 probeExtend;
    float3 probePos;

    if(0 == (gFrameCount - offsetCap) % frameCap) //难道初始的framecount就不为0？但我测了是0
    {
        //uint temp = gFrameCount / frameCap + seed; // 如果没加seed，每次打开程序，种子都和上一次打开是重复的
        dir = generateRandomNormalizedDirection(pixel.x, pixel.y, gSeed);
        probeExtend = generateThreeUniqueRandomNumbers(pixel.x, pixel.y, gSeed);
        probePos = gSceneAABBCenter + probeExtend * gSceneAABBExtent;
        // 本来应该存储probeExtend的，但是因为这里的场景比较小，所以直接存储了probePos
        gOutputProbePoses[pixel] = float4(probePos, 1.0); 
        gOutputRayDirs[pixel] = - float4(dir, 1.0);
    }
    else
    {
        dir = gOutputRayDirs[pixel].xyz;
        probePos = gOutputProbePoses[pixel].xyz;
    }
    //dir = float3(-0.59131, -0.60645, -0.53174);
    //probePos = float3(-0.24109, 0.27686, -0.15479);
    float3 color = MyTracePathRand(pixel, frameDim, dir, probePos);
    gOutputColor[pixel] = float4(color, 1.f);

    //----------------
    
    // //if(!shouldReturnTrue(pixel.x, pixel.y, seed, frameDim)) return;
    // uint2 pixel = DispatchRaysIndex().xy;
    // uint2 frameDim = DispatchRaysDimensions().xy; // frame的大小,这里改成lightProbes的数量
    // float3 color = MyTracePathMap(pixel, frameDim);
    // gOutputColor[pixel] = float4(color, 1.f);

    //----------------


    //uint temp = gFrameCount / 500; // change the output every 5000 frames，这个值在cpp里是加的，怎么还是不会改变渲染结果呢我真的吐了
    //return float3(1.0f);
    //float3 probePos = float3(0.26089, 0.48183, 0.08310);
    //float3 mProbeLoc = float3(0.16089, 0.28183, 0.08310);

    // float radius = 0.005f; 
    // float theta = M_PI * 287 / frameDim.y;
    // float phi = 2 * M_PI * 27 / frameDim.x;
    // float x = sin(theta) * cos(phi);
    // float z = sin(theta) * sin(phi);
    // float y = cos(theta);
    // float3 dir = float3(x, y, z);
    //float3 dir = normalize(float3(x, y, z));

    //下面的逻辑应该五千帧算一次不用老是算，这样上面的frameCount就算不生效也可以在cpp里生成随机种子
    // float3 dir = generateRandomNormalizedDirection(pixel.x, pixel.y, temp);
    // float3 probePos = generateThreeUniqueRandomNumbers(pixel.x, pixel.y, temp);
    // probePos = gSceneAABBCenter + probePos * gSceneAABBExtent;
    // //probePos = float3(0.26089, 0.48183, 0.08310);
    
    // //probePos += gSceneAABBCenter + float3(probePos.x * gSceneAABBExtent.x, probePos.y * gSceneAABBExtent.y, probePos.z * gSceneAABBExtent.z);
    // float3 color = MyTracePathRand(pixel, frameDim, dir, probePos);
    // gOutputProbePoses[pixel] = float4(probePos, 1.0);
    // gOutputRayDirs[pixel] = float4(dir, 1.0);
    
}
