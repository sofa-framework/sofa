// ShaderCustomTex::geometryShaderText

VARYING_IN vec2 texCoord[];
VARYING_IN vec3 Normal[];
VARYING_OUT vec2 fragTexCoord;
VARYING_OUT vec3 N;
void main(void)
{
	//vec3 v1 = POSITION_IN(1).xyz - POSITION_IN(0).xyz;
	//vec3 v2 = POSITION_IN(2).xyz - POSITION_IN(0).xyz;
	//N  = cross(v1,v2);
	//N  = normalize(N);
	 

	int i;
	for(i=0; i< NBVERTS_IN; i++)
	{
		gl_Position = POSITION_IN(i);
		fragTexCoord = texCoord[i];
		N = Normal[i];
		EmitVertex();
	}
	EndPrimitive();
}
