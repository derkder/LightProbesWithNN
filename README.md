UCL CGVI 3rd Term Thesis

Light Probes with NN to deal with high frequency scenes like cautics or shadows

Tring with falcor + tiny cuda nn


---
## Abstract

In path tracing, rendering highly reflective objects is particularly challenging due to the need for rays to bounce multiple times within the scene and accumulate the results to produce the final rendering result. This process often results in noise and slow convergence rates. This thesis presents a novel approach to address these issues by leveraging deep learning to fit a reflection map field. 
    
The concept of the reflection map field is that it is able to generate the reflection map at any position within the scene. To be more specific, for any given position in space, given a ray's direction, the material properties, and the surface normal of this bounce, the proposed network should be able to predict the final rendering result corresponding to the path-traced image, accounting for multiple bounces within the scene.
       
The implementation follows a two-phase approach: In the offline phase, extension data collection is performed by simulating huge amounts of ray-object interactions with random hit information(including the material of the hit surface) within the scene. The inputs and outputs from these simulations are then used to train the deep learning model, effectively learning the reflection map field for the specific scene. During real-time rendering, the trained network is employed to predict the rendered appearance of highly reflective objects, while the traditional path tracing algorithm continues to be used for other surfaces. Extensive experiments under various conditions have validated the effectiveness of the proposed approach, demonstrating significant improvements in rendering quality.

![STEP1](imgs/4_005.png| width=100)
![STEP2](imgs/4_07.png| width=100)


---
## Results
| **(Metallic, Roughness)** | **GT** | **Render Result** | **Render Result (After Sharpen)** |
|---------------------------|--------|-------------------|-----------------------------------|
| **(1.0, 0.0)**             | ![GT](imgs/601.png) | ![Render Result](imgs/602.png) | ![Render Result (After Sharpen)](imgs/603.png) |
| **(0.4, 0.6)**             | ![GT](imgs/610.png) | ![Render Result](imgs/611.png) | ![Render Result (After Sharpen)](imgs/612.png) |
| **(1.0, 0.0)**             | ![GT](imgs/613.png) | ![Render Result](imgs/614.png) | ![Render Result (After Sharpen)](imgs/615.png) |
| **(0.4, 0.6)**             | ![GT](imgs/616.png) | ![Render Result](imgs/617.png) | ![Render Result (After Sharpen)](imgs/618.png) |
| **(1.0, 0.0)**             | ![GT](imgs/619.png) | ![Render Result](imgs/620.png) | ![Render Result (After Sharpen)](imgs/621.png) |
| **(0.4, 0.6)**             | ![GT](imgs/622.png) | ![Render Result](imgs/623.png) | ![Render Result (After Sharpen)](imgs/624.png) |


---
## Report
[Full Report](https://drive.google.com/file/d/1WTEwZddpIXH2OIJBBnEyLT9i7miH33rV/view?usp=drive_link).
