// ShaderColorPerVertex::fragmentShaderText

PRECISION;
VARYING_FRAG vec3 color;
uniform float alpha;
FRAG_OUT_DEF;
void main()
{
#ifdef BLACK_TRANSPARENCY
	if (dot(color,color) == 0.0)
		discard;
#endif
	FRAG_OUT=vec4(color,alpha);
}
