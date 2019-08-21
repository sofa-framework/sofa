#version 330 compatibility

// Input for phong lighting
in vec3 normal;
in vec3 view;

void main(void)
{
	vec3 N = normalize(normal);
	vec3 L = normalize(gl_LightSource[0].position.xyz - view);
	vec3 E = normalize(-view); // eyePos is (0,0,0)
	vec3 R = normalize(-reflect(L, N));

	// Ambient term
	vec4 ambient = gl_FrontLightProduct[0].ambient;

	// Diffuse Term
	vec4 diffuse = gl_FrontLightProduct[0].diffuse * max(dot(N,L), 0.0);
	diffuse = clamp( diffuse, 0.0, 1.0);

	// Specular Term
	vec4 specular = gl_FrontLightProduct[0].specular * pow(max(dot(R,E),0.0), gl_FrontMaterial.shininess);
	specular = clamp(specular, 0.0, 1.0); 

	// Write final color
	gl_FragColor = gl_FrontLightModelProduct.sceneColor + ambient + diffuse + specular;
	//gl_FragColor *= 2;
}
