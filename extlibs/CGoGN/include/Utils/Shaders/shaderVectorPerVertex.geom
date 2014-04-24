// ShaderVectorPerVertex::geometryShaderText

uniform float vectorScale;
uniform mat4 ModelViewProjectionMatrix;
VARYING_IN vec3 VectorAttrib[];
void main()
{
	gl_Position = ModelViewProjectionMatrix * POSITION_IN(0);
	EmitVertex();
	gl_Position = ModelViewProjectionMatrix * (POSITION_IN(0) + vec4(VectorAttrib[0] * vectorScale, 0.0));
	EmitVertex();
	EndPrimitive();
}
