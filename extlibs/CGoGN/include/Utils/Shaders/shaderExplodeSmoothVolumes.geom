// ShaderExplodeSmoothVolumes::geometryShaderText
uniform float explodeV;
uniform float explodeF;
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 NormalMatrix;
uniform mat4 ModelViewMatrix;
uniform vec3 lightPosition;

uniform vec4 plane;

VARYING_IN vec3 colorVertex[4];
VARYING_IN vec3 normalVertex[4];


VARYING_OUT vec3 normalFS;
VARYING_OUT vec3 lightFS;
VARYING_OUT vec3 colorVert;

void main(void)
{
	float d = dot(plane,POSITION_IN(0));
	
	if (d<=0.0)
	{
		for (int i=1; i<=3; i++)
		{
			// explode in face
			vec4 P = explodeF * POSITION_IN(i)  + (1.0-explodeF)* vec4(colorVertex[0],1.0);
			// compute vextex illum from pos & normal 
			vec3 L =  normalize (lightPosition - P.xyz);
			
			vec3 N = normalize (vec3(NormalMatrix*vec4(normalVertex[i],0.0))); 
			
			// explode in volume
			vec4 Q = explodeV *  P + (1.0-explodeV)* POSITION_IN(0);
			gl_Position = ModelViewProjectionMatrix *  Q;

			colorVert = colorVertex[i];
			normalFS = N;
			lightFS = L;
			
			EmitVertex();
		}
		EndPrimitive();
	}
}
