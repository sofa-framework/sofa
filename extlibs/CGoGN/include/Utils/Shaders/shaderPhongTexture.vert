// ShaderPhongTexture::vertexShaderText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexNormal;
ATTRIBUTE vec2 VertexTexCoord;


uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;
uniform mat4 NormalMatrix;
uniform vec3 lightPosition;

VARYING_VERT vec3 EyeVector, Normal, LightDir;
VARYING_VERT vec2 texCoord;

#ifdef WITH_EYEPOSITION
uniform vec3 eyePosition;
#endif

INVARIANT_POS;
void main ()
{
	Normal = vec3 (NormalMatrix * vec4 (VertexNormal, 0.0));
 	vec3 Position = vec3 (ModelViewMatrix * vec4 (VertexPosition, 1.0));
	LightDir = lightPosition - Position;
	#ifdef WITH_EYEPOSITION
		EyeVector = eyePosition-Position;
	#else
		EyeVector = -Position;
	#endif
	texCoord = VertexTexCoord;
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}

