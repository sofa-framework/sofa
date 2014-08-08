varying vec3 normal;
varying vec3 lightDir;

void main()
{
	vec3 NNormal = normalize( normal );
	vec3 NlightDir = normalize( lightDir );
	vec3 ReflectedRay = reflect( NlightDir, NNormal );
	
	vec3 color;
	float dp = clamp( dot( NNormal, NlightDir ), 0., 1. );
	if( dp < 0. ) dp = 0.;
	else if( dp<.5 ) dp = .5;
	else if( dp<.9 ) dp = .9;
	else dp = 1.5;
	
	float dpv = NNormal[2];
	if( dpv>.4 )
		gl_FragColor.xyz = .1 + .9*gl_FrontLightProduct[0].diffuse.xyz * dp;
	else 
	gl_FragColor.xyz = vec3( 0., 0., 0. );
}
