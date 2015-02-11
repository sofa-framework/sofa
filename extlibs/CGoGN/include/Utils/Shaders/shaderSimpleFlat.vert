//ShaderSimpleFlat::vertexShaderText

ATTRIBUTE vec3 VertexPosition, VertexNormal;
#ifdef WITH_COLOR
ATTRIBUTE vec3 VertexColor;
#endif
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;
uniform vec3 lightPosition;
VARYING_VERT vec3 LightDir;
VARYING_VERT vec3 Position;

#ifdef WITH_COLOR
VARYING_VERT vec3 Color;
#endif

INVARIANT_POS;
void main ()
{
	Position = vec3 (ModelViewMatrix * vec4 (VertexPosition, 1.0));
	LightDir = lightPosition - Position;
	#ifdef WITH_COLOR
		Color = VertexColor;
	#endif
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}
