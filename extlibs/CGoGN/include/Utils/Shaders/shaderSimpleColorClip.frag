// ShaderSimpleColor::fragmentShaderClipText

PRECISION;
uniform vec4 color;

uniform vec4 planeClip;
VARYING_FRAG vec3 posClip;

FRAG_OUT_DEF;
void main()
{
	if (dot(planeClip,vec4(posClip,1.0))>0.0)
		discard;

#ifdef BLACK_TRANSPARENCY
	if (dot(color,color) == 0.0)
		discard;
#endif
	FRAG_OUT=color;
}
