// ShaderWallPaper::vertexShaderText

ATTRIBUTE vec2 VertexPosition;
ATTRIBUTE vec2 VertexTexCoord;
VARYING_VERT vec2 texCoord;

uniform vec2 pos;
uniform vec2 sz;

INVARIANT_POS;
void main ()
{
	gl_Position = vec4 (pos+sz*VertexPosition ,0.99999, 1.0);
	texCoord = VertexTexCoord;
}

