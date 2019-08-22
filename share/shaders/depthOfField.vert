#version 120
varying vec3 vertexPosition;
varying vec2 texCoord;

void main()
{
	gl_TexCoord[0] = gl_MultiTexCoord0;
	vertexPosition = (gl_ModelViewMatrix * gl_Vertex).xyz;
	
	texCoord.x = 0.5 * (1.0 + vertexPosition.x);
	texCoord.y = 0.5 * (1.0 + vertexPosition.y);

	gl_Position = ftransform();

} 
