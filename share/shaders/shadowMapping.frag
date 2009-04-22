varying vec3 normal;
varying vec4 ambientGlobal;

uniform int lightFlag[MAX_NUMBER_OF_LIGHTS];
uniform sampler2DShadow shadowTexture[MAX_NUMBER_OF_LIGHTS];

varying vec4 shadowTexCoord[MAX_NUMBER_OF_LIGHTS];
varying vec3 lightDir[MAX_NUMBER_OF_LIGHTS];
varying float dist[MAX_NUMBER_OF_LIGHTS];


void main()
{

	vec4 final_color = ambientGlobal;
	vec4 temp_color;
	bool hasLight = false;
	vec3 n,halfV;
	vec4 diffuse;
	float NdotL,NdotHV;
	float att,spotEffect;
	float isLit;

	//lightFlag[0] = 2;
	//lightFlag[1] = 0;


	// a fragment shader can't write a verying variable, hence we need
	//a new variable to store the normalized interpolated normal

	n = normalize(normal);
	for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ;i++)
	{
		vec4 temp_color = vec4(0.0,0.0,0.0,0.0);
		if(lightFlag[i] > 0)
		{
			hasLight = true;
			isLit=1.0;

			//shadow enabled
			if (lightFlag[i] == 2)
				isLit = shadow2DProj(shadowTexture[i], shadowTexCoord[i]).x;

			NdotL = max(dot(n,normalize(lightDir[i])),0.0);
			if (NdotL > 0.0 && isLit > 0.0)
			{
				diffuse = gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse;
				spotEffect = dot(normalize(gl_LightSource[i].spotDirection), normalize(-lightDir[i]));

				if (spotEffect > gl_LightSource[i].spotCosCutoff)
				{
					spotEffect = isLit * smoothstep(gl_LightSource[i].spotCosCutoff, 1.0, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
					att = spotEffect / (gl_LightSource[i].constantAttenuation +
							gl_LightSource[i].linearAttenuation * dist[i] +
							gl_LightSource[i].quadraticAttenuation * dist[i] * dist[i]);

					temp_color += att * (diffuse * NdotL) ;

					halfV = normalize(gl_LightSource[i].halfVector.xyz);
					NdotHV = max(dot(n,halfV),0.0);
					temp_color += att * gl_FrontMaterial.specular * gl_LightSource[i].specular * pow(NdotHV,gl_FrontMaterial.shininess);
				}
			}
		}

		final_color += temp_color;
	}

	if (hasLight)
		gl_FragColor = final_color;
	else
		gl_FragColor = gl_Color;

}
