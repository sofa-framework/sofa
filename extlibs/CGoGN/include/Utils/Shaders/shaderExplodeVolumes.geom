// ShaderExplodeVolumes::geometryShaderText
uniform float explodeV;
uniform float explodeF;
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 NormalMatrix;
uniform mat4 ModelViewMatrix;
uniform vec3 lightPosition;

uniform vec4 ambient;
uniform vec4 backColor;
uniform vec4 plane;

VARYING_IN vec3 colorVertex[4];
VARYING_OUT vec4 ColorFS;


void main(void)
{
	float d = dot(plane,POSITION_IN(0));
	
	if (d<=0.0)
	{
		vec3 v1 = POSITION_IN(2).xyz - POSITION_IN(1).xyz;
		vec3 v2 = POSITION_IN(3).xyz - POSITION_IN(1).xyz;
		vec3 N  = cross(v1,v2);
		N = normalize (vec3(NormalMatrix*vec4(N,0.0))); 
		
	// compute face center & lighting informations
		vec4 newPos =  ModelViewMatrix * vec4(colorVertex[0],1.0);
		vec3 L =  normalize (lightPosition - newPos.xyz);
		float lambertTerm = dot(N,L);
	
		for (int i=1; i<=3; i++)
		{
			// explode in face
			vec4 P = explodeF * POSITION_IN(i)  + (1.0-explodeF)* vec4(colorVertex[0],1.0);

			// explode in volume
			vec4 Q = explodeV *  P + (1.0-explodeV)* POSITION_IN(0);
			gl_Position = ModelViewProjectionMatrix *  Q;
			if (lambertTerm > 0.0)
				ColorFS = ambient + vec4(colorVertex[i]*lambertTerm, 1.0);
			else
				ColorFS = ambient - backColor*lambertTerm;
			EmitVertex();
		}
		EndPrimitive();
	}
}
