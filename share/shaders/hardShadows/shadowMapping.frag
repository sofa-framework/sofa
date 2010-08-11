varying vec3 normal;
varying vec4 ambientGlobal;

uniform int lightFlag[MAX_NUMBER_OF_LIGHTS];
uniform sampler2DShadow shadowTexture[MAX_NUMBER_OF_LIGHTS];
uniform float zFar[MAX_NUMBER_OF_LIGHTS];
uniform float zNear[MAX_NUMBER_OF_LIGHTS];

varying vec4 shadowTexCoord[MAX_NUMBER_OF_LIGHTS];
varying vec3 lightDir[MAX_NUMBER_OF_LIGHTS];
varying float dist[MAX_NUMBER_OF_LIGHTS];

#ifdef USE_TEXTURE
uniform sampler2D colorTexture;
#endif

void main()
{
	vec4 final_color = ambientGlobal;
    vec4 specular_color = vec4(0.0,0.0,0.0,0.0);
	bool hasLight = false;
	vec3 n,halfV;
	vec4 diffuse;
	float NdotL,NdotHV;
	float att,spotEffect;
	float isLit,shadow;
	float depth, depthSqr;
	//lightFlag[0] = 2;
	//lightFlag[1] = 0;


	// a fragment shader can't write a verying variable, hence we need
	//a new variable to store the normalized interpolated normal

	n = normalize(normal);
	for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ;i++)
	{
		if(lightFlag[i] > 0)
		{
			hasLight = true;
			shadow = 1.0;

			//shadow enabled
			if (lightFlag[i] == 2)
			{
				depthSqr = dot(lightDir[i], lightDir[i]);
				depth = sqrt(depthSqr);
   				depth = (depth - zNear[i])/(zFar[i] - zNear[i]);
				shadow = shadow2DProj(shadowTexture[i], shadowTexCoord[i]).x ;
				if (shadow < depth) shadow = 0.0;
				else shadow = 1.0;				
			}

			NdotL = max(dot(n,normalize(lightDir[i])),0.0);
			if (NdotL > 0.0 && shadow > 0.0)
			{
				diffuse = gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse;
				spotEffect = dot(normalize(gl_LightSource[i].spotDirection), normalize(-lightDir[i]));

				if (spotEffect > gl_LightSource[i].spotCosCutoff)
				{
					spotEffect = shadow * smoothstep(gl_LightSource[i].spotCosCutoff, 1.0, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
					att = spotEffect /* / (gl_LightSource[i].constantAttenuation +
							gl_LightSource[i].linearAttenuation * dist[i] +
							gl_LightSource[i].quadraticAttenuation * dist[i] * dist[i]) */;

					final_color += att * (diffuse * NdotL) ;

					halfV = normalize(gl_LightSource[i].halfVector.xyz);
					NdotHV = max(dot(n,halfV),0.0);
					specular_color += att * gl_FrontMaterial.specular * gl_LightSource[i].specular * pow(NdotHV,gl_FrontMaterial.shininess);
				}
			}
		}
	}

#ifdef USE_TEXTURE
	final_color.rgb *= texture2D(colorTexture,gl_TexCoord[0].st).rgb;
#endif

	final_color.rgb += specular_color.rgb;

	if (hasLight)
		gl_FragColor = final_color;
	else
		gl_FragColor = gl_Color;


	//gl_FragColor = shadow2DProj(shadowTexture[0], shadowTexCoord[0]);
	//gl_FragColor = shadowTexCoord[0]*0.5 - 0.5;

}
