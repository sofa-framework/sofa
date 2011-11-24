#version 120
#extension GL_EXT_gpu_shader4 : enable

void main() 
{
	gl_Position = gl_Vertex;// gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_FrontColor = gl_Color;
}
