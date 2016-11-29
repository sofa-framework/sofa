#version 120

uniform int u_lightType;

varying vec4 lightVec;


void main()
{	
	vec4 lp;
	//lp = lightPosition; 
	//lp = vec4(0.0,0.0,0.0,1.0);

	vec4 ecPos = gl_ModelViewMatrix * gl_Vertex;
	if(u_lightType == 0)
		lightVec = ecPos;//vec4(0.0, 0.0, ecPos.z, 0.0);//vec4(ecPos.xyz - vec3(ecPos.xy,0.0), 0.0);
	else //	if(u_lightType == 2)
		lightVec = ecPos;

	gl_Position = ftransform();
}

