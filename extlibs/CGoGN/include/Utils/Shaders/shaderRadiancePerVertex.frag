// ShaderRadiancePerVertex::fragmentShaderText

PRECISION;
VARYING_FRAG vec3 vxColor;
FRAG_OUT_DEF;
void main (void)
{
		FRAG_OUT = vec4(vxColor,1.0) ;
}
