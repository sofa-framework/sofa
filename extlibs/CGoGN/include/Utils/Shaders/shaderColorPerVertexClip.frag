// ShaderColorPerVertex::fragmentShaderClipText

PRECISION;
VARYING_FRAG vec3 color;
VARYING_FRAG vec3 posClip;

uniform vec4 planeClip;
uniform float alpha;

FRAG_OUT_DEF;

void main()
{
	if (dot(planeClip,vec4(posClip,1.0))>0.0)
		discard;

#ifdef BLACK_TRANSPARENCY
	if (dot(color,color) == 0.0)
		discard;
#endif
	FRAG_OUT=vec4(color,alpha);
}
