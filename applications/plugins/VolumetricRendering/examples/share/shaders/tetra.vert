#version 120

uniform vec4 vertexColor;

varying vec4 volumeColor;
varying vec4 lightDir;
varying mat4 matproj;

void main()
{
	volumeColor = vertexColor; 
	matproj = gl_ModelViewProjectionMatrix;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}