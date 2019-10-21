#version 120

uniform mat4 u_projectionMatrix;
uniform float u_spriteRadius;
uniform float u_spriteScale;

varying vec4 eyeSpacePos;

void main(void)
{
    eyeSpacePos = gl_ModelViewMatrix * gl_Vertex;
    float dist = length(eyeSpacePos);
    gl_PointSize = u_spriteRadius * (u_spriteScale / dist);

    gl_Position = ftransform();
    // //gl_PointSize = u_spriteSize;
    // float tmp = (vec4(u_spriteSize,0,0,0) * u_projectionMatrix / gl_Position.w).x;
    // gl_PointSize = tmp;
    // spriteRadius = tmp*0.5;
}