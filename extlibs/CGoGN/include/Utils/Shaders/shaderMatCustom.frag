//ShaderMatCustom::fragmentShaderText

PRECISON;
VARYING_FRAG vec3 EyeVector, Normal, LightDir;
#ifdef WITH_COLOR
VARYING_FRAG vec3 Color;
#endif
uniform vec4 materialDiffuse;
uniform vec4 materialSpecular;
uniform vec4 materialAmbient;
uniform float shininess;
FRAG_OUT_DEF;
void main()
{	
	vec3 N = normalize (Normal);	
	vec3 L = normalize (LightDir);
	//float lambertTerm = dot(N,L);
	float lambertTerm = 0.15*dot(N,L)+0.80;
	
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
		finalColor += materialDiffuse * lambertTerm;
		#else
		finalColor += vec4((Color*lambertTerm),0.0) ;
		#endif
		vec3 E = normalize(EyeVector);
		vec3 R = reflect(-L, N);
		float specular = pow( max(dot(R, E), 0.0), shininess );
		finalColor += materialSpecular * specular;
	}
	FRAG_OUT=finalColor;
	//FRAG_OUT = vec4(lambertTerm,lambertTerm,lambertTerm,0);
}
