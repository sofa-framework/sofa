// ShaderFlat::geometryShaderText

uniform float explode;
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 NormalMatrix;
uniform mat4 ModelViewMatrix;
uniform vec3 lightPosition;
VARYING_OUT float lambertTerm;

void main(void)
{
	vec3 v1 = POSITION_IN(1).xyz - POSITION_IN(0).xyz;
	vec3 v2 = POSITION_IN(2).xyz - POSITION_IN(0).xyz;
	vec3 N  = cross(v1,v2);
	N  =  normalize(vec3(NormalMatrix*vec4(N,0.0)));

	vec3 center = POSITION_IN(0).xyz + POSITION_IN(1).xyz + POSITION_IN(2).xyz; 
	center /= 3.0;
	vec4 newPos =  ModelViewMatrix * vec4(center,1.0);
	vec3 L =  normalize (lightPosition - newPos.xyz);
	lambertTerm = clamp(dot(N,L),0.0,1.0);

	int i;
	for(i=0; i< NBVERTS_IN; i++)
	{
		vec4 pos =  explode * POSITION_IN(i) + (1.0-explode)* vec4(center,1.0);
		gl_Position = ModelViewProjectionMatrix *  pos;
		EmitVertex();
	}
	EndPrimitive();
}
