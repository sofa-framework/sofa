// ShaderSimpleColor::vertexShaderClipText

ATTRIBUTE vec3 VertexPosition, VertexNormal;
uniform mat4 ModelViewProjectionMatrix;

VARYING_VERT vec3 posClip;

void main ()
{
	posClip = VertexPosition;
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}
