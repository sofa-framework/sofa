#version 120

uniform int lightFlag[MAX_NUMBER_OF_LIGHTS];

varying vec3 normal;
varying vec4 ambientGlobal;
varying vec3 lightDir[MAX_NUMBER_OF_LIGHTS];
varying float dist[MAX_NUMBER_OF_LIGHTS];
varying float spotOff[MAX_NUMBER_OF_LIGHTS];

#ifdef USE_TEXTURE
uniform sampler2D colorTexture;
#endif // USE_TEXTURE

#if ENABLE_SHADOW == 1 
uniform sampler2D shadowTexture[MAX_NUMBER_OF_LIGHTS];
uniform float zFar[MAX_NUMBER_OF_LIGHTS];
uniform float zNear[MAX_NUMBER_OF_LIGHTS];
varying vec4 shadowTexCoord[MAX_NUMBER_OF_LIGHTS];
#endif // ENABLE_SHADOW == 1 

#if ENABLE_SHADOW == 1 
float g_MinVariance = 0.01;
float g_bleedingAmount = 0.2;

float ChebyshevUpperBound(vec2 moments, float t)
{
    // One-tailed inequality valid if t > Moments.x
    float p = float(t <= moments.x);

    // Compute variance.
    float variance = moments.y - moments.x*moments.x;
    variance = max(variance, g_MinVariance);

    // Compute probabilistic upper bound.
    float d = t - moments.x;
    float p_max = variance / (variance + d*d);
    return max(p, p_max);
}

float ReduceLightBleeding(float p_max, float amount)
{
    // Remove the [0, amount] tail and linearly rescale (amount, 1].
    return smoothstep(amount, 1.0, p_max);
}

float VarianceShadow(vec2 moments, float depth)
{
    float p_max = ChebyshevUpperBound(moments, depth);
    p_max = ReduceLightBleeding(p_max, g_bleedingAmount);
    return p_max;
}

float shadow_variance(sampler2D shadowMap, vec4 shadowCoord, float depth) 
{ 
	return VarianceShadow(texture2DProj(shadowMap, shadowCoord).xy, depth);
}

float shadow_variance_unroll(const int index, const sampler2D shadowTexture) 
{ 
	float depth, depthSqr;
	depthSqr = dot(lightDir[index], lightDir[index]);
	depth = sqrt(depthSqr);
	depth = (depth - zNear[index])/(zFar[index] - zNear[index]);

	return VarianceShadow(texture2DProj(shadowTexture, shadowTexCoord[index]).xy, depth);
}
#endif // ENABLE_SHADOW == 1 

void main()
{
	vec4 final_color = ambientGlobal;
   	vec4 specular_color = vec4(0.0,0.0,0.0,0.0);
	bool hasLight = false;
	vec3 n,halfV;
	vec4 diffuse;
	float NdotL,NdotHV;
	float att,spotEffect;

#if ENABLE_SHADOW == 1 
	float shadow=1.0;
	
	//Compute first all shadow variables
	//as : -Mac Os does not support accessing array of sampler2D with a non const index
	//     -Linux does not support accessing more than 1 time an element in an array (at compilation time)
	       
	       
#if MAX_NUMBER_OF_LIGHTS > 0
	 float shadowsVar[MAX_NUMBER_OF_LIGHTS];
	 for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; i++)
	 	shadowsVar[i] = 1.0;
	 	
	 if(lightFlag[0] == 2)
	 	shadowsVar[0] = shadow_variance_unroll(0, shadowTexture[0]);
#if MAX_NUMBER_OF_LIGHTS > 1
	 if(lightFlag[1] == 2)
	 	shadowsVar[1] = shadow_variance_unroll(1, shadowTexture[1]);
#endif //MAX_NUMBER_OF_LIGHTS > 1
#if MAX_NUMBER_OF_LIGHTS > 2
	 if(lightFlag[2] == 2)
	 	shadowsVar[2] = shadow_variance_unroll(2, shadowTexture[2]);
#endif //MAX_NUMBER_OF_LIGHTS > 2
#if MAX_NUMBER_OF_LIGHTS > 3
	 if(lightFlag[3] == 2)
	 	shadowsVar[3] = shadow_variance_unroll(3, shadowTexture[3]);
#endif //MAX_NUMBER_OF_LIGHTS > 3
#if MAX_NUMBER_OF_LIGHTS > 4
	 if(lightFlag[4] == 2)
	 	shadowsVar[4] = shadow_variance_unroll(4, shadowTexture[4]);
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

			NdotL = max(dot(n,normalize(lightDir[i])),0.0);	
			if (NdotL > 0.0)
			{
#if ENABLE_SHADOW == 1 
				if (shadow > 0.0)
				{
#endif // ENABLE_SHADOW == 1 
					diffuse = gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse;
					spotEffect = dot(normalize(gl_LightSource[i].spotDirection), normalize(-lightDir[i]));

					if (spotEffect > spotOff[i])
					{
						spotEffect = smoothstep(spotOff[i], 1.0, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
#if ENABLE_SHADOW == 1 
						spotEffect *= shadow;
#endif // ENABLE_SHADOW == 1 
						att = spotEffect /* / (gl_LightSource[i].constantAttenuation +
								gl_LightSource[i].linearAttenuation * dist[i] +
								gl_LightSource[i].quadraticAttenuation * dist[i] * dist[i]) */;

						final_color += att * (diffuse * NdotL) ;

						halfV = normalize(gl_LightSource[i].halfVector).xyz;
						NdotHV = max(dot(n,halfV),0.0);
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

	//gl_FragColor = vec4(shadowsVar[0], shadowsVar[0], shadowsVar[0], 1.0);
}


