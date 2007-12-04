//phong

varying vec3 normal;
varying vec3 lightDir;
varying vec4 SunCoord;
varying vec3 viewVector;
uniform vec3 lightpos;
uniform vec3 campos;

void main()
{
	SunCoord = gl_TextureMatrix[0] * gl_Vertex;
	vec4 pos = gl_ModelViewMatrix * gl_Vertex;
	//vec3 lightpos = gl_LightSource[0].position.xyz;
	//lightpos = gl_ModelViewMatrix * vec4( lightpos, 1 );
	lightDir =  normalize( lightpos - gl_Vertex.xyz );
	//viewVector = normalize( gl_Vertex.xyz - gl_ModelViewMatrix[3].xyz );
	viewVector = -( campos - gl_Vertex.xyz );
	
	normal = gl_Normal;
	gl_Position = ftransform();
}
