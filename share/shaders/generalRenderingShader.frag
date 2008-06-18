
#ifdef TEXTURE_UNIT_0
uniform sampler2D colorTexture;
#endif

#ifdef PLANE_ENVIRONMENT_MAPPING
varying vec3 normalVec;
varying vec3 viewVec;
uniform sampler2D planeTexture;
uniform float altitude;
//uniform int axis; //0 = +X, 1 = +Y, 2 = +Z, 3 = -X, 4 = -Y, 5 = -Z
uniform float border_gamma, border_alpha;
#endif

#ifdef PHONG
varying vec3 lightDir /* , halfVector */ , normalView;
#endif

varying vec4 diffuse, ambient, specular;

vec3 reflect(vec3 I, vec3 N)
{
  return I - 2.0 * N * dot(N, I);
}

void main()
{
	vec4 color = gl_Color;
	
#ifdef TEXTURE_UNIT_0
	color.rgb = texture2D(colorTexture,gl_TexCoord[0].st).rgb;
#endif
	
#ifdef PHONG	
	//Phong
	vec3 n;
	float NdotL,NdotHV;
	//float att,spotEffect;
	vec4 phong_color = ambient;
	
	/* a fragment shader can't write a verying variable, hence we need
	a new variable to store the normalized interpolated normal */
	n = normalize(normalView);
	
	/* compute the dot product between normal and ldir */
	NdotL = max(dot(n,normalize(lightDir)),0.0);

	if (NdotL > 0.0) {
	
		/*spotEffect = dot(normalize(gl_LightSource[0].spotDirection), normalize(-lightDir));
		if (spotEffect > gl_LightSource[0].spotCosCutoff) {
			spotEffect = pow(spotEffect, gl_LightSource[0].spotExponent);
			att = spotEffect / (gl_LightSource[0].constantAttenuation +
					gl_LightSource[0].linearAttenuation * dist +
					gl_LightSource[0].quadraticAttenuation * dist * dist);*/
				
			phong_color += /*att * */ (diffuse * NdotL) ;
			
		//}
	}
	color.rgb *= phong_color.rgb;
	//end phong
	
	
#endif

	// Perform a simple 2D texture look up.
	//vec3 base_color = gl_Color.xyz;//texture2D(planeTexture, reflectVec.xz).rgb;

#ifdef PLANE_ENVIRONMENT_MAPPING
	vec3 unitNormalVec = normalize(normalVec);
	vec3 unitViewVec = normalize(viewVec);
	color.a = color.a + (border_alpha - color.a)* (pow( 1.0 - abs(dot(unitNormalVec,unitViewVec)), border_gamma));

	color.rgb *= color.a;

	vec3 reflectVec = reflect(viewVec, normalVec);
	
	if (reflectVec.z>0.0)
	{
		// Perform a cube map look up.
	  color.rgb += texture2D(planeTexture, reflectVec.xy*( altitude/reflectVec.z )+vec2(0.5,0.5)).rgb * specular.rgb;

	}
#endif


	// Write the final pixel.
	gl_FragColor = color;

}
