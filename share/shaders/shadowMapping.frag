varying vec3 normal;

varying vec4 ambientGlobal;
#ifdef SHADOW_LIGHT0
uniform int shadowActive0;
uniform sampler2DShadow shadowTexture0;
varying vec4 shadowTexCoord0;
varying vec4 diffuse;
varying vec3 lightDir,halfVector;
varying float dist;
#endif

#ifdef SHADOW_LIGHT1
uniform int shadowActive1;
uniform sampler2DShadow shadowTexture1;
varying vec4 shadowTexCoord1;
varying vec4 diffuse1;
varying vec3 lightDir1,halfVector1;
varying float dist1;
#endif


void main()
{

	vec4 final_color = ambientGlobal;
	bool hasLight = false;
	vec3 n,halfV;
	float NdotL,NdotHV;
	float att,spotEffect;
	float isLit;


	/* a fragment shader can't write a verying variable, hence we need
	a new variable to store the normalized interpolated normal */
	n = normalize(normal);

#ifdef SHADOW_LIGHT0
	hasLight = true;
	vec4 color = vec4(0.0,0.0,0.0,0.0);

	/* compute the dot product between normal and ldir */
	NdotL = max(dot(n,normalize(lightDir)),0.0);
	isLit = shadow2DProj(shadowTexture0, shadowTexCoord0).x;
	if (shadowActive0 == 0)
		isLit=1.0;

	if (NdotL > 0.0 && isLit > 0.0) {

		spotEffect = dot(normalize(gl_LightSource[0].spotDirection), normalize(-lightDir));
		if (spotEffect > gl_LightSource[0].spotCosCutoff) {
			spotEffect = isLit * smoothstep(gl_LightSource[0].spotCosCutoff, 1.0, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
			att = spotEffect / (gl_LightSource[0].constantAttenuation +
					gl_LightSource[0].linearAttenuation * dist +
					gl_LightSource[0].quadraticAttenuation * dist * dist);

			color += att * (diffuse * NdotL) ;

			halfV = normalize(halfVector);
			NdotHV = max(dot(n,halfV),0.0);
			color += att * gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(NdotHV,gl_FrontMaterial.shininess);
		}
	}
	final_color += color;
#endif
#ifdef SHADOW_LIGHT1
	hasLight = true;
	vec4 color1 = vec4(0.0,0.0,0.0,0.0);

	/* compute the dot product between normal and ldir */
	NdotL = max(dot(n,normalize(lightDir1)),0.0);
	isLit = shadow2DProj(shadowTexture1, shadowTexCoord1).x;
	if (shadowActive1 == 0)
			isLit=1.0;

	if (NdotL > 0.0 && isLit > 0.0) {

		spotEffect = dot(normalize(gl_LightSource[1].spotDirection), normalize(-lightDir1));
		if (spotEffect > gl_LightSource[1].spotCosCutoff) {
			spotEffect = isLit * smoothstep(gl_LightSource[1].spotCosCutoff, 1.0, spotEffect); //pow(spotEffect, gl_LightSource[1].spotExponent);
			att = spotEffect / (gl_LightSource[1].constantAttenuation +
					gl_LightSource[1].linearAttenuation * dist1 +
					gl_LightSource[1].quadraticAttenuation * dist1 * dist1);

			color1 += att * (diffuse1 * NdotL) ;

			halfV = normalize(halfVector1);
			NdotHV = max(dot(n,halfV),0.0);
			color1 += att * gl_FrontMaterial.specular * gl_LightSource[1].specular * pow(NdotHV,gl_FrontMaterial.shininess);
		}
	}
	final_color += color1;
#endif
	if (hasLight)
		gl_FragColor = final_color;
	else gl_FragColor = gl_Color;


}
