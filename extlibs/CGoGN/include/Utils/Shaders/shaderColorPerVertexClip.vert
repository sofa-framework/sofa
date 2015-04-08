// ShaderColorPerVertex::vertexShaderClipText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexColor;

uniform mat4 ModelViewProjectionMatrix;

VARYING_VERT vec3 color;
VARYING_VERT vec3 posClip;

INVARIANT_POS;
void main ()
{
	posClip = VertexPosition;
	color = VertexColor;
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}
