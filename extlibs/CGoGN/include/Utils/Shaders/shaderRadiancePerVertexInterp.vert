// ShaderRadiancePerVertex::vertexShaderInterpText

ATTRIBUTE vec3 VertexPosition;
ATTRIBUTE vec3 VertexNormal;
ATTRIBUTE ivec2 VertexParam;

uniform mat4 ModelViewProjectionMatrix ;

VARYING_VERT vec3 vnorm;
VARYING_VERT vec3 vpos;
VARYING_VERT ivec2 vtexcoord ;

INVARIANT_POS;

void main ()
{
	vpos = VertexPosition;
	vnorm = VertexNormal;
	vtexcoord = VertexParam;

    gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}
