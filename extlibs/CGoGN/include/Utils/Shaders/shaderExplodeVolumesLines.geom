// ShaderExplodeVolumesLines::geometryShaderText
uniform float explodeV;
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 NormalMatrix;
uniform mat4 ModelViewMatrix;
uniform vec4 plane;
uniform vec4 color;

VARYING_OUT vec4 ColorFS;
void main(void)
{
	float d = dot(plane,POSITION_IN(0));
	
	if (d<=0.0)
	{
		ColorFS = color;
	
		for (int i=1; i<NBVERTS_IN; i++)
		{
			vec4 P = explodeV * POSITION_IN(i) + (1.0-explodeV)* POSITION_IN(0);
			gl_Position = ModelViewProjectionMatrix *  P;
			EmitVertex();
		}
		EndPrimitive();
	}	
}
