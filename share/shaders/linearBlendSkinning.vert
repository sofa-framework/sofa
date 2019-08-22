/*************************************************************
GLSL - Linear Blend Skinning using displacement matrices.
(c) 2015 Francois Faure, Anatoscope
*************************************************************/

attribute vec4 indices;
attribute vec4 weights;

// Linear blend skinning inputs
#define MAX_BONES 64
uniform mat4 boneMatrix[MAX_BONES];


// Light
varying vec3 normal;
varying vec3 view;


void main()
{
        vec4 pos = boneMatrix[int(indices[0])] * gl_Vertex * weights[0];
        pos += boneMatrix[int(indices[1])] * gl_Vertex * weights[1];
        pos += boneMatrix[int(indices[2])] * gl_Vertex * weights[2];
        pos += boneMatrix[int(indices[3])] * gl_Vertex * weights[3];

	gl_Position = gl_ModelViewProjectionMatrix * pos;
	
	//-------------------------------------------------------
	
	vec4 normal4 = vec4(gl_Normal.xyz, 0.0);
        vec4 norm = boneMatrix[int(indices[0])] * normal4 * weights[0];
        norm += boneMatrix[int(indices[1])] * normal4 * weights[1];
        norm += boneMatrix[int(indices[2])] * normal4 * weights[2];
        norm += boneMatrix[int(indices[3])] * normal4 * weights[3];
	
	view = vec3(gl_ModelViewMatrix * pos);
	normal = gl_NormalMatrix*norm.xyz;
}
