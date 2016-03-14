#version 120

// uniform vec4 vertexColor;
//GLSL >= 130
//in vec4 a_vertexColor;
attribute vec4 a_vertexColor;

//GLSL >= 130
//out vec4 volumeColor;
//out vec4 lightDir;
//out mat4 matproj;
varying vec4 volumeColor;
varying vec4 lightDir;
varying mat4 matproj;

void main()
{
	vec4 u_plane0 = vec4(0, 0, 1, 0);

	volumeColor = a_vertexColor; 
	matproj = gl_ModelViewProjectionMatrix;

	//gl_ClipDistance[0] = -1; //dot(gl_ModelViewMatrix * gl_Vertex, u_plane0);
	gl_ClipVertex = gl_ModelViewMatrix* gl_Vertex;;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

}