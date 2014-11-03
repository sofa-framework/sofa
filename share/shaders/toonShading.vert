varying vec3 normal;
varying vec3 lightDir;

void main()
{
	vec4 pos = gl_ModelViewMatrix * gl_Vertex;
	vec3 lightpos = gl_LightSource[0].position.xyz;
	lightDir =  lightpos - pos.xyz;
	normal = gl_NormalMatrix * gl_Normal;
	gl_Position = ftransform();
}
