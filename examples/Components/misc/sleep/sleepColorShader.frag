varying vec3 normal;
varying vec3 lightDir;
uniform vec3 color;

void main()
{
	vec3 NNormal = normalize( normal );
	vec3 NlightDir = normalize( lightDir );
	float dp = clamp( dot( NNormal, NlightDir ), 0., 1. );
	
	// gl_FrontLightProduct[0].diffuse.xyz
	gl_FragColor.xyz = color * dp;
}
