//ShaderPhong::vertexShaderClipText

ATTRIBUTE vec3 VertexPosition, VertexNormal;
#ifdef WITH_COLOR
ATTRIBUTE vec3 VertexColor;
#endif
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;
uniform mat4 NormalMatrix;
uniform vec3 lightPosition;
VARYING_VERT vec3 EyeVector, Normal, LightDir;

#ifdef WITH_COLOR
VARYING_VERT vec3 Color;
#endif

#ifdef WITH_EYEPOSITION
uniform vec3 eyePosition;
#endif

VARYING_VERT vec3 posClip;

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
	#ifdef WITH_COLOR
		Color = VertexColor;
	#endif

	posClip = VertexPosition;
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}
