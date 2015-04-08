// PointSprite::vertexShaderText 

ATTRIBUTE vec3 VertexPosition;
#ifdef WITH_COLOR_PER_VERTEX 
	ATTRIBUTE vec4 VertexColor;
	VARYING_VERT vec4 color;
#endif

VARYING_VERT vec3 posClip;

void main ()
{
	posClip = VertexPosition;
	gl_Position = vec4(VertexPosition, 1.0);
#ifdef WITH_COLOR_PER_VERTEX 
	color = VertexColor;
#endif
}
