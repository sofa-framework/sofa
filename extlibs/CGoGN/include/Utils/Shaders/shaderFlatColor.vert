// ShaderFlatColor::vertexShaderText
ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexColor;
VARYING_VERT vec3 colorVertex;
void main()
{
	gl_Position = vec4(VertexPosition, 1.0);
	colorVertex = VertexColor;
}
