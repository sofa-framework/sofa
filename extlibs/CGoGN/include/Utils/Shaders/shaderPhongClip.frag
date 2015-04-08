//ShaderPhong::fragmentShaderClipText

PRECISION;
VARYING_FRAG vec3 EyeVector, Normal, LightDir;
#ifdef WITH_COLOR
VARYING_FRAG vec3 Color;
#endif
uniform vec4 materialDiffuse;
uniform vec4 materialSpecular;
uniform vec4 materialAmbient;
uniform float shininess;
uniform vec4 backColor;

uniform vec4 planeClip;
VARYING_FRAG vec3 posClip;

FRAG_OUT_DEF;

void main()
{
	if (dot(planeClip,vec4(posClip,1.0))>0.0)
		discard;

	vec3 N = normalize (Normal);
	vec3 L = normalize (LightDir);
	//float lambertTerm = clamp(dot(N,L),0.0,1.0);

	vec4 finalColor = materialAmbient;

#ifdef DOUBLE_SIDED
	float lambertTerm;
	vec4 diffuseColor = materialDiffuse;
	if (!gl_FrontFacing)
	{
		N *= -1.0;
		lambertTerm = clamp(dot(N,L),0.0,1.0);
	}
	else
		lambertTerm = clamp(dot(N,L),0.0,1.0);
#ifndef WITH_COLOR
	finalColor += materialDiffuse * lambertTerm;
#else
	finalColor += vec4((Color*lambertTerm),0.0) ;
#endif
	vec3 E = normalize(EyeVector);
	vec3 R = reflect(-L, N);
	float specular = pow( max(dot(R, E), 0.0), shininess );
	finalColor += materialSpecular * specular;
#else
	float lambertTerm = clamp(dot(N,L),0.0,1.0);
	if (gl_FrontFacing)
	{
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
	else
	{
		finalColor = backColor;
	}
#endif
	FRAG_OUT=finalColor;
}
