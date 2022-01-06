# Pathtracer
<img src="https://github.com/RiceLiao/TDA362-Pathtracer/blob/main/img/ship.jpg"> | <img src="https://github.com/RiceLiao/TDA362-Pathtracer/blob/main/img/ball.jpg">
:-------------------------:|:-------------------------:

# Summary
The path-tracer is implemented in TDA362 Computer Graphics at Chalmers University of Technology. It is a CPU-version path-tracer rendering photorealistic images. I mainly focus on how to generate physically plausible images, and the ray-tracing part will be done by Intel's "Embree" library. It includes multiple linearly blended BRDFs with importance sampling.

## Refraction
The refraction is implemented by blending the glass material. So that a glass ball that correctly refracts its surroundings is seen.