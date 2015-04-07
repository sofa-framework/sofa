// ShaderVectorPerVertex::vertexShaderText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexVector;
VARYING_VERT vec3 VectorAttrib;
VARYING_VERT vec3 posClip;

INVARIANT_POS;
void main ()
{
	posClip = VertexPosition;
	VectorAttrib = VertexVector;
	gl_Position = vec4(VertexPosition, 1.0);
}
