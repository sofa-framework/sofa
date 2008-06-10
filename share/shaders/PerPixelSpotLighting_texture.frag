varying vec4 diffuse,ambientGlobal, ambient;
varying vec3 normal,lightDir,halfVector;
varying float dist;
varying vec3 lightAxis;

uniform sampler2D colorTexture;

void main()
{
	
	vec4 ctex = texture2D(colorTexture, gl_TexCoord[0].st);

	vec3 n,halfV;
	float NdotL,NdotHV;
	vec4 color = ambientGlobal*ctex;
	float att,spotEffect;
	
	/* a fragment shader can't write a verying variable, hence we need
	a new variable to store the normalized interpolated normal */
	n = normalize(normal);
	
	/* compute the dot product between normal and ldir */
	NdotL = max(dot(n,normalize(lightDir)),0.0);

	if (NdotL > 0.0) {
	
	
		vec3 temp = gl_LightSource[0].spotDirection;
		//temp = lightAxis;
		
		spotEffect = dot(normalize(temp), normalize(-lightDir));
		if (spotEffect > 0.0) {
			spotEffect = pow(spotEffect, gl_LightSource[0].spotExponent) ;
			att = spotEffect / (gl_LightSource[0].constantAttenuation +
					gl_LightSource[0].linearAttenuation * dist +
					gl_LightSource[0].quadraticAttenuation * dist * dist);
				
			color += att * (diffuse * NdotL + ambient) * ctex ;
			
			halfV = normalize(halfVector);
			NdotHV = max(dot(n,halfV),0.0);
			color += att * gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(NdotHV,gl_FrontMaterial.shininess);
		}

	}
	gl_FragColor = color;
}
