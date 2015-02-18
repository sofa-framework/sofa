// ShaderSimpleColor::fragmentShaderText

PRECISON;
uniform vec4 color;
FRAG_OUT_DEF;
void main()
{
#ifdef BLACK_TRANSPARENCY
	if (dot(color,color) == 0.0)
		discard;
#endif
	FRAG_OUT=color;
}
