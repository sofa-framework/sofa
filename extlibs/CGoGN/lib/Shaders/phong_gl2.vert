//Vertex_Shader

attribute vec3 ColorPerVertex;

varying vec3 normal, lightDir, eyeVec;
varying vec3 vcolor; 

void main()
{	
	normal = gl_NormalMatrix * gl_Normal;

	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

	lightDir = vec3(gl_LightSource[0].position.xyz - vVertex);
	eyeVec = -vVertex;

	vcolor = ColorPerVertex;

	gl_Position = ftransform();		
}
