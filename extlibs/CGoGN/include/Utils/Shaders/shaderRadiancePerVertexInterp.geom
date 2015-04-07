// ShaderRadiancePerVertex::geometryShaderInterpText

VARYING_IN vec3 vpos[];
VARYING_IN vec3 vnorm[];
VARYING_IN ivec2 vtexcoord[];

VARYING_OUT vec3 vxPos;
VARYING_OUT vec3 vxNorm;
VARYING_OUT vec3 barycentricCoord;

flat out ivec2 vx0TexCoord;
flat out ivec2 vx1TexCoord;
flat out ivec2 vx2TexCoord;

void main()
{
	vx0TexCoord = vtexcoord[0];
	vx1TexCoord = vtexcoord[1];
	vx2TexCoord = vtexcoord[2];

	gl_Position = POSITION_IN(0);
	vxPos = vpos[0];
	vxNorm = vnorm[0];
	barycentricCoord = vec3(1., 0., 0.);
	EmitVertex();

	gl_Position = POSITION_IN(1);
	vxPos = vpos[1];
	vxNorm = vnorm[1];
	barycentricCoord = vec3(0., 1., 0.);
	EmitVertex();

	gl_Position = POSITION_IN(2);
	vxPos = vpos[2];
	vxNorm = vnorm[2];
	barycentricCoord = vec3(0., 0., 1.);
	EmitVertex();

	EndPrimitive();
}
