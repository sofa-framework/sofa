// ShaderVectorPerVertex::geometryShaderText

uniform float vectorScale;
uniform mat4 ModelViewProjectionMatrix;
VARYING_IN vec3 VectorAttrib[];

uniform vec4 planeClip;
VARYING_IN vec3 posClip[];

void main()
{
	if (dot(planeClip,vec4(posClip[0],1.0))<=0.0)
	{
		gl_Position = ModelViewProjectionMatrix * POSITION_IN(0);
		EmitVertex();
		gl_Position = ModelViewProjectionMatrix * (POSITION_IN(0) + vec4(VectorAttrib[0] * vectorScale, 0.0));
		EmitVertex();
		EndPrimitive();
	}
}
