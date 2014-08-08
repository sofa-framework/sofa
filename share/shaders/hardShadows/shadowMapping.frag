#version 120

varying vec3 normal;
varying vec4 ambientGlobal;

uniform int lightFlag[MAX_NUMBER_OF_LIGHTS];
varying vec3 lightDir[MAX_NUMBER_OF_LIGHTS];
//varying float dist[MAX_NUMBER_OF_LIGHTS];
//varying float spotOff[MAX_NUMBER_OF_LIGHTS];

#ifdef USE_TEXTURE
uniform sampler2D colorTexture;
#endif // USE_TEXTURE

#if ENABLE_SHADOW == 1 
uniform sampler2DShadow shadowTexture[MAX_NUMBER_OF_LIGHTS];
uniform float zFar[MAX_NUMBER_OF_LIGHTS];
uniform float zNear[MAX_NUMBER_OF_LIGHTS];
varying vec4 shadowTexCoord[MAX_NUMBER_OF_LIGHTS];

float shadow_unroll(const int index, const sampler2DShadow shadowTexture) 
{ 
	float depth, depthSqr;
	depthSqr = dot(lightDir[index], lightDir[index]);
	depth = sqrt(depthSqr);
	depth = (depth - zNear[index])/(zFar[index] - zNear[index]);
	float shadow = shadow2DProj(shadowTexture, shadowTexCoord[index]).x;
	if (shadow+0.005f < depth)
		shadow = 0.0f;
	else 
		shadow = 1.0f;	
	return shadow;
}
#endif // ENABLE_SHADOW == 1 

void main()
{
	vec4 final_color = ambientGlobal;
    vec4 specular_color = vec4(0.0f,0.0f,0.0f,0.0f);
	bool hasLight = false;
	vec3 n,halfV;
	vec4 diffuse;
	float NdotL,NdotHV;
	float att,spotEffect;
#if ENABLE_SHADOW == 1 
	float shadow, depth, depthSqr;

	//Compute first all shadow variables
	//as : -Mac Os does not support accessing array of sampler2D with a non const index
	//     -Linux does not support accessing more than 1 time an element in an array (at compilation time)
	       
	       
#if MAX_NUMBER_OF_LIGHTS > 0
	 float shadowsVar[MAX_NUMBER_OF_LIGHTS];
	 for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; i++)
	 	shadowsVar[i] = 1.0f;
	 	
	 if(lightFlag[0] == 2)
	 	shadowsVar[0] = shadow_unroll(0, shadowTexture[0]);
#if MAX_NUMBER_OF_LIGHTS > 1
	 if(lightFlag[1] == 2)
	 	shadowsVar[1] = shadow_unroll(1, shadowTexture[1]);
#endif //MAX_NUMBER_OF_LIGHTS > 1
#if MAX_NUMBER_OF_LIGHTS > 2
	 if(lightFlag[2] == 2)
	 	shadowsVar[2] = shadow_unroll(2, shadowTexture[2]);
#endif //MAX_NUMBER_OF_LIGHTS > 2
#if MAX_NUMBER_OF_LIGHTS > 3
	 if(lightFlag[3] == 2)
	 	shadowsVar[3] = shadow_unroll(3, shadowTexture[3]);
#endif //MAX_NUMBER_OF_LIGHTS > 3
#if MAX_NUMBER_OF_LIGHTS > 4
	 if(lightFlag[4] == 2)
	 	shadowsVar[4] = shadow_unroll(4, shadowTexture[4]);
#endif //MAX_NUMBER_OF_LIGHTS > 4

#endif //MAX_NUMBER_OF_LIGHTS > 0

#endif // ENABLE_SHADOW == 1 
	// a fragment shader can't write a verying variable, hence we need
	//a new variable to store the normalized interpolated normal

	n = normalize(normal);
	for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ;i++)
	{
		if(lightFlag[i] > 0)
		{
			hasLight = true;
#if ENABLE_SHADOW == 1 
			shadow = shadowsVar[i];
#endif // ENABLE_SHADOW == 1 
			NdotL = max(dot(n,normalize(lightDir[i])),0.0f);
			if (NdotL > 0.0f)
			{
#if ENABLE_SHADOW == 1 
				if (shadow > 0.0f)
				{
#endif // ENABLE_SHADOW == 1 
					diffuse = gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse;
					spotEffect = dot(normalize(gl_LightSource[i].spotDirection), normalize(-lightDir[i]));
                    float spotOff = gl_LightSource[i].spotCosCutoff;
					if (spotEffect > spotOff)
					{
						spotEffect = smoothstep(spotOff, 1.0f, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
#if ENABLE_SHADOW == 1 
						spotEffect *= shadow;
#endif // ENABLE_SHADOW == 1 

                        float dist = length(lightDir[i]);
						att = spotEffect /* / (gl_LightSource[i].constantAttenuation +
								gl_LightSource[i].linearAttenuation * dist +
								gl_LightSource[i].quadraticAttenuation * dist * dist) */;

						final_color += att * (diffuse * NdotL) ;

						halfV = normalize(gl_LightSource[i].halfVector.xyz);
						NdotHV = max(dot(n,halfV),0.0f);
						specular_color += att * gl_FrontMaterial.specular * gl_LightSource[i].specular * pow(NdotHV,gl_FrontMaterial.shininess);
					}
				#if ENABLE_SHADOW == 1 
				}
#endif // ENABLE_SHADOW == 1 
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

}
