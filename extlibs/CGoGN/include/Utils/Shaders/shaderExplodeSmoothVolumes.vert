// ShaderExplodeSmoothVolumes::vertexShaderText
ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexNormal;
ATTRIBUTE vec3 VertexColor;

VARYING_VERT vec3 normalVertex;
VARYING_VERT vec3 colorVertex;

void main()
{
	colorVertex = VertexColor;
	normalVertex = VertexNormal;
	gl_Position = vec4(VertexPosition, 1.0);
}
