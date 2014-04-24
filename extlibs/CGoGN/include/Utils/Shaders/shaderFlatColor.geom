// ShaderFlatColor::geometryShaderText
uniform float explode;
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 NormalMatrix;
uniform mat4 ModelViewMatrix;
uniform vec3 lightPosition;
uniform vec4 ambient;

VARYING_IN vec3 colorVertex[3]; // input is triangles -> 3
VARYING_OUT vec4 ColorFS;

void main(void)
{
	vec3 v1 = POSITION_IN(1).xyz - POSITION_IN(0).xyz;
	vec3 v2 = POSITION_IN(2).xyz - POSITION_IN(0).xyz;
	vec3 N  = cross(v1,v2);
	N  =  normalize (vec3(NormalMatrix*vec4(N,0.0))); 
	vec3 center = POSITION_IN(0).xyz + POSITION_IN(1).xyz + POSITION_IN(2).xyz; 
	center /= 3.0;
	vec4 newPos =  ModelViewMatrix * vec4(center,0.0);
	vec3 L =  normalize (lightPosition - newPos.xyz);
	float lambertTerm = dot(N,L);

#ifdef AVERAGE_COLOR		
	int i;
	vec3 averageColor=vec3(0.,0.,0.);
	for(i=0; i< 3; i++)
		averageColor +=colorVertex[i];
	averageColor /= 3.0;
	
	ColorFS = ambient;
	if(lambertTerm > 0.0)
		ColorFS += vec4(averageColor,1.0) * lambertTerm;
			
	for(i=0; i< 3; i++)
	{
		vec4 pos =  explode * POSITION_IN(i) + (1.0-explode)* vec4(center,1.0);
		gl_Position = ModelViewProjectionMatrix *  pos;	
		EmitVertex();
	}
	EndPrimitive();
#else
	int i;
	for(i=0; i< 3; i++)
	{
		vec4 pos =  explode * POSITION_IN(i) + (1.0-explode)* vec4(center,1.0);
		ColorFS = ambient;
		if(lambertTerm > 0.0)
			ColorFS += vec4(colorVertex[i],1.0) * lambertTerm;
		gl_Position = ModelViewProjectionMatrix *  pos;
		EmitVertex();
	}
	EndPrimitive();
#endif

}
