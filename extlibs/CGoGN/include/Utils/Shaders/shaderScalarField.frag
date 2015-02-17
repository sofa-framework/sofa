// ShaderScalarField::fragmentShaderText

PRECISON;
uniform float minValue;
uniform float maxValue;
VARYING_FRAG vec3 color;
VARYING_FRAG float scalar;
FRAG_OUT_DEF;

void main()
{
//	float s = scalar * 10.0;
//	if( s - floor(s) <= 0.05 )
//		FRAG_OUT = vec4(0.0);
//	else
		FRAG_OUT = vec4(color, 0.0);
}
