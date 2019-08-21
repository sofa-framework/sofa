#version 130

// uniform vec4 vertexColor;
in vec4 a_vertexColor;

//GLSL >= 130
out vec4 volumeColor;
out vec4 lightDir;
out mat4 matproj;
//varying out vec4 volumeColor;
//varying out vec4 lightDir;
//varying out mat4 matproj;

void main()
{
	vec4 u_plane0 = vec4(0, 0, 1, 0);

	volumeColor = a_vertexColor; 
	matproj = gl_ModelViewProjectionMatrix;

	//gl_ClipDistance[0] = -1; //dot(gl_ModelViewMatrix * gl_Vertex, u_plane0);
	gl_ClipVertex = gl_ModelViewMatrix* gl_Vertex;;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

}