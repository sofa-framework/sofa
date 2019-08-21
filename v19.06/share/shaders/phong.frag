//phong

varying vec3 normal;
varying vec3 lightDir;
uniform sampler2D ShadowMap;
varying vec4 SunCoord;
uniform float SunZNear;
uniform float SunZFar;
varying vec3 viewVector;

void main()
{
	vec3 DiffuseColor = gl_FrontLightProduct[0].diffuse.rgb;
	vec3 AmbientColor = gl_FrontLightProduct[0].ambient.rgb;
	vec3 SpecColor = gl_FrontLightProduct[0].specular.rgb;
	
	vec3 Nnormal = normalize( normal );
	vec3 NlightDir = normalize( lightDir );
	vec3 ReflectedRay = reflect( NlightDir, Nnormal );
	vec3 NViewVector = normalize( viewVector );
	
	
	float RealSunDepth = ( SunCoord.z - SunZNear )/( SunZFar-SunZNear );
	vec2 coord = SunCoord.xy/SunCoord.w*.5 + .5;
	float SunDepth = texture2D( ShadowMap, coord  ).r;
	if( coord.x<0 || coord.x>1. ) SunDepth=RealSunDepth;
	if( coord.y<0 || coord.y>1. ) SunDepth=RealSunDepth;
	if( SunDepth == 1. ) SunDepth=RealSunDepth;

	
	vec3 color = SpecColor * vec3( clamp( pow( dot( ReflectedRay, NViewVector ), 10. ), 0., 1. ) );
	color += DiffuseColor * clamp( dot( Nnormal, NlightDir ), 0., 1. );
	vec3 Lit = .8 * color;
	vec3 Ambient = DiffuseColor;
	
	if( RealSunDepth <= SunDepth+2./256. )
	{
		gl_FragColor.rgb = Ambient + Lit;
	}
	else
	{
		gl_FragColor.rgb = Ambient;
	}
	gl_FragColor.a = 1.;
}

