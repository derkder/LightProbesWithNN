#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct CommonParams {
    uint2   frameDim = { 0, 0 };
    uint    frameCount = 0;
};

struct BasicQuery {
    float4 hitPos;
    float4 rayDir;
};

struct AdvancedQuery {
    float4 hitPos;
    float4 rayDir;
    float4 diffuse;
};

END_NAMESPACE_FALCOR
