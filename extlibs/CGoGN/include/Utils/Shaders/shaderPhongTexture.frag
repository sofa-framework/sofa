// ShaderPhongTexture::fragmentShaderText

PRECISON;
VARYING_FRAG vec3 EyeVector, Normal, LightDir;
VARYING_FRAG vec2 texCoord;

uniform vec4 materialDiffuse;
uniform vec4 materialSpecular;
uniform float ambientCoef;
uniform float shininess;
uniform sampler2D textureUnit;

FRAG_OUT_DEF;

void main()
{
	vec3 N = normalize (Normal);
	vec3 L = normalize (LightDir);
	float lambertTerm = dot(N,L);

	vec4 finalColor = ambientCoef * texture2D(textureUnit,texCoord);
	
	#ifdef DOUBLE_SIDED
	if (lambertTerm < 0.0)
	{
		N = -1.0*N;
		lambertTerm = -1.0*lambertTerm;
	#else
	if (lambertTerm > 0.0)
	{
	#endif
		vec3 E = normalize(EyeVector);
		vec3 R = reflect(-L, N);
		float specular = pow( max(dot(R, E), 0.0), shininess );
		vec3 diffuse = (1.0 - ambientCoef) * texture2D(textureUnit,texCoord).rgb;
		finalColor += vec4(diffuse*lambertTerm,0.0) + materialSpecular*specular;
	}
	FRAG_OUT=finalColor;
}
