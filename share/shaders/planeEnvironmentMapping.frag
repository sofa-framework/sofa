varying vec3 normalVec;
varying vec3 viewVec;

uniform sampler2D planeTexture;
uniform float altitudeY;

varying vec4 diffuse,ambientGlobal, ambient, specular;
varying vec3 lightDir,halfVector,normalView;
varying float dist;


vec3 reflect(vec3 I, vec3 N)
{
  return I - 2.0 * N * dot(N, I);
}

void main()
{
	float reflect_factor = 1.0;
	
	//Phong
	vec3 n,halfV;
	float NdotL,NdotHV;
	vec4 color = ambientGlobal + ambient;
	float att,spotEffect;
	
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
				
			color += /*att * */ (diffuse * NdotL) ;
			
		//}
	}
	
	//end phong
	
	
	// Perform a simple 2D texture look up.
	vec3 base_color = gl_Color.xyz;//texture2D(planeTexture, reflectVec.xz).rgb;
	
	vec3 cube_color = vec3(0.0,0.0,0.0);
	
	vec3 reflectVec = reflect(viewVec, normalVec);
	
	if (reflectVec.y>0.0)
	{
	
		float t = altitudeY/reflectVec.y;
	
		// Perform a cube map look up.
        //cube_color = vec3(reflectVec.xz*t+vec2(0.5,0.5),1.0);
        cube_color = texture2D(planeTexture, reflectVec.xz*t+vec2(0.5,0.5)).rgb;
        cube_color *= specular.xyz;

	}

	// Write the final pixel.
	gl_FragColor = vec4(color.xyz+cube_color,1.0); //vec4( mix(base_color, cube_color, reflect_factor), 1.0);
}
