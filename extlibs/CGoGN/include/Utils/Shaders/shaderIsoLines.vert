// ShaderIsoLines::vertexShaderText
ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE float VertexData;
VARYING_VERT float attribData;
void main()
{
	gl_Position = vec4(VertexPosition, 1.0);
	attribData = VertexData;	
}
