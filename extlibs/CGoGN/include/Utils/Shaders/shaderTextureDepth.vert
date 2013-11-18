// ShaderTextureDepth::vertexShaderText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec2 VertexTexCoord;
uniform mat4 ModelViewProjectionMatrix;
VARYING_VERT vec2 texCoord;
INVARIANT_POS;
void main ()
{
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
	texCoord = VertexTexCoord;
}

