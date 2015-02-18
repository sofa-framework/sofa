//ShaderEnvMap::fragmentShaderText

PRECISON;
VARYING_FRAG vec3 EyeVector, Normal, LightDir;
#ifdef WITH_COLOR
VARYING_FRAG vec3 Color;
#endif
uniform vec4 materialDiffuse;
uniform vec4 materialAmbient;
uniform samplerCube EnvMap;
uniform float blendCoef;

FRAG_OUT_DEF;
void main()
{
	vec3 N = normalize (Normal);
	vec3 L = normalize (LightDir);
	float lambertTerm = dot(N,L);

	vec4 finalColor = materialAmbient;
	
	#ifdef DOUBLE_SIDED
	if (lambertTerm < 0.0)
	{
		N = -1.0*N;
		lambertTerm = -1.0*lambertTerm;
	#else
	if (lambertTerm > 0.0)
	{
	#endif
		#ifndef WITH_COLOR
		vec4 col = materialDiffuse;
		#else
		vec4 col = vec4(Color,0.0) ;
		#endif
		
		vec3 R  = reflect(-EyeVector,N);
		finalColor += mix(col,textureCube(EnvMap,R),blendCoef) * lambertTerm;
	}	
	FRAG_OUT=finalColor;
}
