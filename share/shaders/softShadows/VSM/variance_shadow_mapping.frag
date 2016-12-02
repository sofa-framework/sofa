#version 120

uniform int u_lightFlags[MAX_NUMBER_OF_LIGHTS];
uniform int u_lightTypes[MAX_NUMBER_OF_LIGHTS];

#ifdef USE_TEXTURE
uniform sampler2D u_colorTexture;
#endif // USE_TEXTURE

uniform float u_shadowFactors[MAX_NUMBER_OF_LIGHTS];
uniform float u_lightBleedings[MAX_NUMBER_OF_LIGHTS];
uniform float u_minVariances[MAX_NUMBER_OF_LIGHTS];

varying vec3 normal;
varying vec4 ambientGlobal;
varying vec3 lightDirs[MAX_NUMBER_OF_LIGHTS];
varying vec4 lightSpacePosition[MAX_NUMBER_OF_LIGHTS];

#if ENABLE_SHADOW == 1 
uniform sampler2D u_shadowTextures[MAX_NUMBER_OF_LIGHTS];
uniform float u_zFars[MAX_NUMBER_OF_LIGHTS];
uniform float u_zNears[MAX_NUMBER_OF_LIGHTS];

varying vec4 shadowTexCoord[MAX_NUMBER_OF_LIGHTS];


float linearDepth(float depth, float zNear, float zFar)
{
	float z = depth;      // fetch the z-value from our depth texture
	float n = zNear;                                // the near plane
	float f = zFar;                               // the far plane
	float c = (2.0 * n) / (f + n - z * (f - n));  // convert to linear values 
	 
	return c;                      // linear
}

float ChebyshevUpperBound(const int index, vec2 moments, float t)
{
    // One-tailed inequality valid if t > Moments.x
    float p = float(t <= moments.x);

    // Compute variance.
    float variance = moments.y - moments.x*moments.x;
    variance = max(variance, u_minVariances[index]);

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

float VarianceShadow(const int index, vec2 moments, float depth)
{
    float p_max = ChebyshevUpperBound(index, moments, depth);
    p_max = ReduceLightBleeding(p_max, u_lightBleedings[index]);
    return p_max;
}

float computeShadowFactor(const int index) 
{ 
	float shadow = 1.0;

	if(u_lightTypes[index] == 0)
	{		
		vec3 proj = lightSpacePosition[index].xyz / lightSpacePosition[index].w;
	    vec2 uv;
	    uv.x = 0.5 * proj.x + 0.5;
	    uv.y = 0.5 * proj.y + 0.5;
	    float z = (0.5 * proj.z + 0.5); // linear because orthograhic projection
	    vec2 moments = texture2D(u_shadowTextures[index], uv).xy; // linear

		shadow = VarianceShadow(index, moments, z);
	}
	else if(u_lightTypes[index] == 2)
	{
		vec3 proj = lightSpacePosition[index].xyz / lightSpacePosition[index].w;
	    vec2 uv;
	    uv.x = 0.5 * proj.x + 0.5;
	    uv.y = 0.5 * proj.y + 0.5;
	    float z = 0.5 * proj.z + 0.5;  //not linear because of perspective projection 
	    z = linearDepth(z, u_zNears[index], u_zFars[index])*0.5; // so make it linear, but why /2 ????

	    vec2 moments = texture2D(u_shadowTextures[index], uv).xy; // linear

		shadow = VarianceShadow(index, moments, z);
	}

	return shadow;
}
#endif // ENABLE_SHADOW == 1 


vec4 computeSpotLightShading(const int index, vec3 n, float NdotL, float shadow)  
{
	vec3 final_color = vec3(0.0, 0.0, 0.0);
	vec3 specular_color = vec3(0.0, 0.0, 0.0);
	vec3 diffuse = (gl_FrontMaterial.diffuse * gl_LightSource[index].diffuse).xyz;


	float spotEffect = dot(normalize(gl_LightSource[index].spotDirection), normalize(-lightDirs[index]));
	float spotOff = gl_LightSource[index].spotCosCutoff;
	if (spotEffect > spotOff)
	{
		spotEffect = smoothstep(spotOff, 1.0f, spotEffect); 
		spotEffect = pow(spotEffect, gl_LightSource[index].spotExponent);
		float shadowSpotEffect = spotEffect;
#if ENABLE_SHADOW == 1 
		if(shadow < 1 )
			shadowSpotEffect *=  (shadow + (1 - u_shadowFactors[index]) );

		if(shadowSpotEffect > spotEffect)	
			shadowSpotEffect = spotEffect;

#endif // ENABLE_SHADOW == 1 

		float att = shadowSpotEffect ;

		final_color += att * (diffuse * NdotL) ;

		vec3 halfV = normalize(gl_LightSource[index].halfVector.xyz);
		float NdotHV = max(dot(n,halfV),0.0f);
		specular_color += att * gl_FrontMaterial.specular.rgb * gl_LightSource[index].specular.rgb * pow(NdotHV,gl_FrontMaterial.shininess);
	}

#ifdef USE_TEXTURE
	final_color.rgb *= texture2D(u_colorTexture,gl_TexCoord[0].st).rgb;
#endif

	final_color.rgb += specular_color.rgb;

	return vec4(final_color, 1.0);
}


vec4 computeDirectionalShading(const int index, vec3 n, float NdotL, float shadow) 
{
	vec3 final_color = vec3(0.0, 0.0, 0.0);
	vec3 specular_color = vec3(0.0, 0.0, 0.0);
	vec3 diffuse = (gl_FrontMaterial.diffuse * gl_LightSource[index].diffuse).xyz;

	final_color = (diffuse * NdotL);

	if(shadow < 0.99 )
		final_color *=  (shadow + (1 - u_shadowFactors[index]) );

	vec3 halfV = normalize(gl_LightSource[index].halfVector.xyz);
	float NdotHV = max(dot(n,halfV),0.0f);
	specular_color += gl_FrontMaterial.specular.rgb * gl_LightSource[index].specular.rgb * pow(NdotHV,gl_FrontMaterial.shininess);

#ifdef USE_TEXTURE
	final_color.rgb *= texture2D(u_colorTexture,gl_TexCoord[0].st).rgb;
#endif

	//final_color.rgb += specular_color.rgb;

	return vec4(final_color, 1.0);
}




void main()
{
	vec4 final_color = ambientGlobal;
	bool hasLight = false;
	vec3 n;
	float NdotL;
	float shadow = 1.0;
#if ENABLE_SHADOW == 1 
	//Compute first all shadow variables
	//as : -Mac Os does not support accessing array of sampler2D with a non const index
	//     -Linux does not support accessing more than 1 time an element in an array (at compilation time)
	       
	       
#if MAX_NUMBER_OF_LIGHTS > 0
	 float shadowsVar[MAX_NUMBER_OF_LIGHTS];
	 for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; i++)
	 	shadowsVar[i] = 1.0f;
	 	
	 if(u_lightFlags[0] == 2)
	 	shadowsVar[0] = computeShadowFactor(0);
#if MAX_NUMBER_OF_LIGHTS > 1
	 if(u_lightFlags[1] == 2)
	 	shadowsVar[1] = computeShadowFactor(1);
#endif //MAX_NUMBER_OF_LIGHTS > 1
#if MAX_NUMBER_OF_LIGHTS > 2
	 if(u_lightFlags[2] == 2)
	 	shadowsVar[2] = computeShadowFactor(2);
#endif //MAX_NUMBER_OF_LIGHTS > 2
#if MAX_NUMBER_OF_LIGHTS > 3
	 if(u_lightFlags[3] == 2)
	 	shadowsVar[3] = computeShadowFactor(3);
#endif //MAX_NUMBER_OF_LIGHTS > 3
#if MAX_NUMBER_OF_LIGHTS > 4
	 if(u_lightFlags[4] == 2)
	 	shadowsVar[4] = computeShadowFactor(4);
#endif //MAX_NUMBER_OF_LIGHTS > 4

#endif //MAX_NUMBER_OF_LIGHTS > 0

#endif // ENABLE_SHADOW == 1 
	// a fragment shader can't write a verying variable, hence we need
	//a new variable to store the normalized interpolated normal

	n = normalize(normal);
	for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ;i++)
	{
		if(u_lightFlags[i] > 0)
		{
			hasLight = true;
#if ENABLE_SHADOW == 1 
			shadow = shadowsVar[i];
#endif // ENABLE_SHADOW == 1 
			NdotL = max(dot(n,normalize(lightDirs[i])),0.0f);
			if (NdotL > 0.0f)
			{
				//if (shadow > 0.0f)
				{
					//SpotLight
					if(u_lightTypes[i] == 2)
					{
						final_color += computeSpotLightShading(i, n, NdotL, shadow);
					}
					//Directional
					else if(u_lightTypes[i] == 0)
					{
						final_color += computeDirectionalShading(i, n, NdotL, shadow);
					}
				}
			}
		}
	}

	if (hasLight)
		gl_FragColor = final_color;
	else
		gl_FragColor = gl_Color;

// 					float t =  0.0;
// #if ENABLE_SHADOW == 1 
// 	 t = shadowsVar[0];
// #endif // ENABLE_SHADOW == 1 
// 	gl_FragColor = vec4(t,t,t,1);
// 	//gl_FragColor = vec4(lightSpacePosition[0].xyz*10, 1.0);

}



