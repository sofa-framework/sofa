#version 120

varying vec3 normal, lightDir, eyeVec;
//varying float att;


void main (void)
{
	vec4 final_color = 
	(gl_FrontLightModelProduct.sceneColor * gl_FrontMaterial.ambient) + 
	(gl_LightSource[0].ambient * gl_FrontMaterial.ambient);
							
	vec3 N = normalize(normal);
	vec3 L = normalize(lightDir);
	
	float lambertTerm = dot(N,L);
	
	if(lambertTerm > 0.0)
	{
		final_color += gl_LightSource[0].diffuse * 
		gl_FrontMaterial.diffuse * 
		lambertTerm ;//* att;	
		
		
		// RED REFLEX EFFECT
		#ifdef REDREFLEX
		float redReflexFactor; // gradient value for redReflex intensity
		vec3 redReflexColor = vec3(0.9, 0.29, 0);
		float rrlimitAngle = 0.5; // define the red reflex tolerance for normal.light dot product
		if(lambertTerm > rrlimitAngle)
		{
			redReflexFactor = (lambertTerm-rrlimitAngle) * (1.0/(1-rrlimitAngle));
			
			final_color.xyz = redReflexColor*redReflexFactor + final_color.xyz*(1.0-redReflexFactor);

		}
		#endif
		// ******************
		
		vec3 E = normalize(eyeVec); // camera vector
		vec3 R = reflect(-L, N); // reflected light vector
		
		float specular = pow( max(dot(R, E), 0.0), gl_FrontMaterial.shininess );
		
		final_color += gl_LightSource[0].specular * 
		gl_FrontMaterial.specular * specular ;//* att;	
	}

	gl_FragColor = final_color;			
}