// ShaderVectorPerVertex::vertexShaderText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexVector;
VARYING_VERT vec3 VectorAttrib;
INVARIANT_POS;
void main ()
{
	VectorAttrib = VertexVector;
	gl_Position = vec4(VertexPosition, 1.0);
}
