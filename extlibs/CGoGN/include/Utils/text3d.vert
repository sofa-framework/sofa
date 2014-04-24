//Strings3D::vertexShaderText

ATTRIBUTE vec4 VertexPosition;
uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform vec3 strPos;
uniform float scale;
VARYING_VERT vec2 tex_coord;
INVARIANT_POS;

#ifdef WITH_PLANE
uniform vec3 planeX;
uniform vec3 planeY;
#endif

void main ()
{
#ifdef WITH_PLANE
	vec4 pos = ModelViewMatrix * vec4(strPos,1.0);
	pos += scale*VertexPosition[0]*vec4(planeX,0.0);
	pos += scale*VertexPosition[1]*vec4(planeY,0.0);
#else
	vec4 pos = ModelViewMatrix * vec4(strPos,1.0) + vec4(VertexPosition[0]*scale,VertexPosition[1]*scale,0.0,0.0);
#endif

	tex_coord = vec2(VertexPosition[2],VertexPosition[3]);
	gl_Position = ProjectionMatrix * pos;
}

