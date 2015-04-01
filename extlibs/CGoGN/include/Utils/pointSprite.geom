// PointSprite::geometryShaderText

uniform float size;
uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform vec4 planeClip;

VARYING_IN vec3 posClip[1];

#ifdef WITH_PLANE
uniform vec3 eyePos;
VARYING_OUT vec3 shiftedEye;
#endif

VARYING_OUT vec2 spriteCoord;
VARYING_OUT vec3 sphereCenter;

#ifdef WITH_COLOR_PER_VERTEX 
VARYING_IN vec4 color[1];
VARYING_OUT vec4 colorsprite;
#endif

#ifdef WITH_PLANE
void corner( vec4 center, vec3 planeX, vec3 planeY, float x, float y)
{
	spriteCoord = vec2(x,y);
	vec4 pos = center + size*( x*vec4(planeX,0.0) + y*vec4(planeY,0.0)+ vec4(0.0,0.0,0.5,0.0));
	gl_Position = ProjectionMatrix *  pos;
	EmitVertex();
}
#else
void corner( vec4 center, float x, float y)
{
	spriteCoord = vec2(x,y);
	vec4 pos = center + vec4(size*x, size*y, 0.0, 0.0);
	gl_Position = ProjectionMatrix *  pos;
	EmitVertex();
}
#endif

void main()
{
	if (dot(planeClip,vec4(posClip[0],1.0))<=0.0)
	{
#ifdef WITH_COLOR_PER_VERTEX 
	colorsprite = color[0];
#endif
	vec4 posCenter = ModelViewMatrix * POSITION_IN(0);
	sphereCenter = posCenter.xyz;
	
#ifdef WITH_PLANE
	shiftedEye = eyePos - sphereCenter;
	vec3 V = -shiftedEye;
	normalize(V);

	vec3 planeX = vec3(-V[2],0.0,V[0]); //cross(V, vec3(0.0,1.0,0.0));
	normalize(planeX);
	vec3 planeY = cross(planeX,V);

	corner(posCenter, planeX, planeY, -1.4, 1.4);
	corner(posCenter, planeX, planeY, -1.4,-1.4);
	corner(posCenter, planeX, planeY,  1.4, 1.4);
	corner(posCenter, planeX, planeY,  1.4,-1.4);
#else
	corner(posCenter, -1.4, 1.4);
	corner(posCenter, -1.4,-1.4);
	corner(posCenter,  1.4, 1.4);
	corner(posCenter,  1.4,-1.4);
#endif	

	EndPrimitive();
	}
}

