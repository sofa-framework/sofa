#version 120

uniform vec4 vertexColor;

varying vec4 tetraColor;
varying vec4 lightDir;

void main()
{
	tetraColor = vertexColor; 
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}