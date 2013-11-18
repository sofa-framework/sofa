// ShaderColorPerVertex::vertexShaderText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexColor;
uniform mat4 ModelViewProjectionMatrix;
VARYING_VERT vec3 color;
INVARIANT_POS;
void main ()
{
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
	color = VertexColor;
}
