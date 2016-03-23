uniform float u_enableLight;

//GLSL >= 130
//in vec3 triangleNormal;
varying vec3 triangleNormal;

void main()
{
	vec4 finalColor = vec4(1,1,1,1);

	if(u_enableLight < 1.0)
	{
		finalColor = gl_Color;
	}
	else
	{
		vec4 diffuse = gl_Color;

		//light comes from the camera
		vec3 lightDir = vec3(0,0,1);  
		vec4 lightColor = vec4(1,1,1,1);

		vec3 norm = normalize(triangleNormal);

		float diff = max(abs(dot(norm, lightDir)), 0.0);
		vec4 diffuseComp = diff * lightColor;

		finalColor = diffuse * diffuseComp;
		//finalColor = vec4(norm, 1.0);
	}

	gl_FragColor = finalColor;
}