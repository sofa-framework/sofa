varying vec3 normal;
varying vec4 ambientGlobal;

uniform int lightFlag[MAX_NUMBER_OF_LIGHTS];
uniform sampler2D shadowTexture[MAX_NUMBER_OF_LIGHTS];
uniform float zFar[MAX_NUMBER_OF_LIGHTS];
uniform float zNear[MAX_NUMBER_OF_LIGHTS];

varying vec4 shadowTexCoord[MAX_NUMBER_OF_LIGHTS];
varying vec3 lightDir[MAX_NUMBER_OF_LIGHTS];
//varying float dist[MAX_NUMBER_OF_LIGHTS];

#ifdef USE_TEXTURE
uniform sampler2D colorTexture;
#endif

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

float shadow_variance(int index, float depthSqr) 
{ 
	vec4 shadowTexCoord = shadowTexCoord[index];
	vec2 moment;
	
	//Array of Sampler2D can only be accessed with a constant
	//with mac os x.
	if (index == 0)
		moment = texture2DProj(shadowTexture[0], shadowTexCoord).xy;
	if (index == 1)
		moment = texture2DProj(shadowTexture[1], shadowTexCoord).xy;
#if MAX_NUMBER_OF_LIGHTS > 2
	if (index == 2)
		moment = texture2DProj(shadowTexture[2], shadowTexCoord).xy;
#endif
#if MAX_NUMBER_OF_LIGHTS > 3
	if (index == 3)
		moment = texture2DProj(shadowTexture[3], shadowTexCoord).xy;
#endif
#if MAX_NUMBER_OF_LIGHTS > 4
	if (index == 4)
		moment = texture2DProj(shadowTexture[4], shadowTexCoord).xy;
#endif		
	return VarianceShadow(moment, depthSqr);
}


void main()
{
	vec4 final_color = ambientGlobal;
   	vec4 specular_color = vec4(0.0,0.0,0.0,0.0);
	bool hasLight = false;
	vec3 n,halfV;
	vec4 diffuse;
	float NdotL,NdotHV;
	float att,spotEffect;

	//lightFlag[0] = 2;
	//lightFlag[1] = 0;
	float shadow=1.0;
	float depth, depthSqr;

	// a fragment shader can't write a verying variable, hence we need
	//a new variable to store the normalized interpolated normal

	n = normalize(normal);
	
	for(int i=0 ; i<MAX_NUMBER_OF_LIGHTS ;i++)
	{
		int flag = lightFlag[i]; 
		if(flag > 0)
		{
			hasLight = true;
			depthSqr = dot(lightDir[i], lightDir[i]);
			depth = sqrt(depthSqr);
   			depth = (depth - zNear[i])/(zFar[i] - zNear[i]);
			shadow = 1.0;

			if(flag == 2)
				shadow = shadow_variance(i, depth);

			NdotL = max(dot(n,normalize(lightDir[i])),0.0);	
			if (shadow > 0.0 && NdotL > 0.0)
			{
				
				diffuse = gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse;
				spotEffect = dot(normalize(gl_LightSource[i].spotDirection), normalize(-lightDir[i]));

				if (spotEffect > gl_LightSource[i].spotCosCutoff)
				{

					spotEffect = (shadow)*smoothstep(gl_LightSource[i].spotCosCutoff, 1.0, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
					att = spotEffect /* / (gl_LightSource[i].constantAttenuation +
							gl_LightSource[i].linearAttenuation * dist[i] +
							gl_LightSource[i].quadraticAttenuation * dist[i] * dist[i]) */;

					final_color += att * (diffuse * NdotL) ;

					halfV = normalize(gl_LightSource[i].halfVector).xyz;
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

	//gl_FragColor = vec4(shadow, shadow, shadow, 1.0);
}


