#version 120

uniform mat4 u_projectionMatrix;
uniform mat4 u_modelviewMatrix;
// uniform float u_spriteRadius;
// uniform float u_spriteScale;

// varying vec4 eyeSpacePos;
// varying float ndcDepth;
// varying float spriteRadius;


void main(void)
{
	//float radius = 1.0;
	//eyeSpacePos = gl_ModelViewMatrix * gl_Vertex;
	//screenSpacePos = u_projectionMatrix * eyeSpacePos;

	//float dist = length(eyeSpacePos);
    gl_PointSize = 2.0;//u_spriteRadius * (u_spriteScale / dist);

    gl_Position = u_projectionMatrix * u_modelviewMatrix * gl_Vertex;
    // //gl_PointSize = u_spriteSize;
    // float tmp = (vec4(u_spriteSize,0,0,0) * u_projectionMatrix / gl_Position.w).x;
    // gl_PointSize = tmp;
    // spriteRadius = tmp*0.5;
}
