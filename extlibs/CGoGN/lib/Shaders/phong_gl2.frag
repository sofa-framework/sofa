//Pixel_Shader

varying vec3 normal, lightDir, eyeVec;
varying vec3 vcolor; 

void main (void)
{
	vec4 final_color = vec4(0.0,0.0,0.0,0.0);
							
	vec3 N = normalize(normal);
	vec3 L = normalize(lightDir);
	
	float lambertTerm = dot(N,L);
	
	if(lambertTerm > 0.0)
	{
		final_color += vec4(vcolor,1.0) * lambertTerm;	
		
		vec3 E = normalize(eyeVec);
		vec3 R = reflect(-L, N);
		float specular = pow( max(dot(R, E), 0.0), 
		                 gl_FrontMaterial.shininess );
		final_color += gl_FrontMaterial.specular * 
					   specular;	
	}

	gl_FragColor = final_color;

}
