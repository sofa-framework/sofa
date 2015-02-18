// ShaderExplodeSmoothVolumes::fragmentShaderText

uniform vec4 ambient;

VARYING_FRAG vec3 normalFS;
VARYING_FRAG vec3 lightFS;
VARYING_FRAG vec3 colorVert;
FRAG_OUT_DEF;

void main()
{
	float lambertTerm = abs(dot(normalize(normalFS),normalize(lightFS)));
	FRAG_OUT = ambient + vec4(colorVert*lambertTerm, 1.0);
}
