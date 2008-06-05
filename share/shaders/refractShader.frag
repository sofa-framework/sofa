varying vec3 normalVec;
varying vec3 viewVec;

uniform sampler2D planeTexture;
uniform float altitude;
uniform int axis; //0 = +X, 1 = +Y, 2 = +Z, 3 = -X, 4 = -Y, 5 = -Z
uniform float border_gamma, border_alpha;

varying vec4 diffuse,ambientGlobal, ambient, specular;
varying vec3 lightDir,halfVector,normalView;
varying float dist;

varying vec3 positionW;
uniform vec3 worldSphereCenter;
uniform vec3 lightCenterProj;
uniform vec3 discCenter;

uniform float sphereRadius;
uniform float refractCoeff;

uniform float lightProjRadius;


vec3 reflect(vec3 I, vec3 N)
{
  return I - 2.0 * N * dot(N, I);
}

vec3 refract2(vec3 I, vec3 N, float etaRatio)
{
  float cosI = dot(-I, N);
  float cosT2 = 1.0 - etaRatio * etaRatio *
                       (1.0 / cosI * cosI);
  if (cosT2 > 0.0)
  return  etaRatio * I +
             ((etaRatio * cosI - sqrt(abs(cosT2))) * N);
  else
  	return vec3(0.0,0.0,0.0);
  
}


void main()
{
	float reflect_factor = 1.0;
	
	//Phong
	vec3 n,halfV;
	float NdotL,NdotHV;
	vec4 color = ambientGlobal + ambient;
	float att,spotEffect;
	
	/* a fragment shader can't write a verying variable, hence we need
	a new variable to store the normalized interpolated normal */
	n = normalize(normalView);
	
	/* compute the dot product between normal and ldir */
	NdotL = max(dot(n,normalize(lightDir)),0.0);

	if (NdotL > 0.0) {
	
		/*spotEffect = dot(normalize(gl_LightSource[0].spotDirection), normalize(-lightDir));
		if (spotEffect > gl_LightSource[0].spotCosCutoff) {
			spotEffect = pow(spotEffect, gl_LightSource[0].spotExponent);
			att = spotEffect / (gl_LightSource[0].constantAttenuation +
					gl_LightSource[0].linearAttenuation * dist +
					gl_LightSource[0].quadraticAttenuation * dist * dist);*/
				
			color += /*att * */ (diffuse * NdotL) ;
			
		//}
	}
	
	//end phong
	color = gl_Color;
	
	// Perform a simple 2D texture look up.
	vec3 base_color = gl_Color.xyz;//texture2D(planeTexture, reflectVec.xz).rgb;
	
	vec3 cube_color = vec3(0.0,0.0,0.0);
	
	vec3 reflectVec = reflect(viewVec, normalVec);
	
	bool testAxis = false;
	float t = 0.0;
	
	vec2 subReflectVec = vec2(0.0,0.0);
	
	if (axis == 0){
		testAxis = (reflectVec.x>0.0);
		t = altitude/reflectVec.x;
		subReflectVec = reflectVec.yz;
	}
	
	if (axis == 1){
		testAxis = (reflectVec.y>0.0);
		t = altitude/reflectVec.y;
		subReflectVec = reflectVec.xz;
	}
	if (axis == 2){
		testAxis = (reflectVec.z>0.0);
		t = altitude/reflectVec.z;
		subReflectVec = reflectVec.xy;
	}
	
	if (testAxis)
	{
		// Perform a cube map look up.
        cube_color = texture2D(planeTexture, subReflectVec*t+vec2(0.5,0.5)).rgb;
        cube_color *= specular.xyz;

	}
	vec3 unitNormalVec = normalize(normalVec);
	vec3 unitViewVec = normalize(viewVec);
	float alpha_color = color.w;
	float alpha_color2 = alpha_color + (border_alpha - alpha_color)* (pow( 1.0 - abs(dot(unitNormalVec,unitViewVec)), border_gamma));
	
	// Write the final pixel.
	//gl_FragColor = vec4((color.xyz*alpha_color2)+cube_color,alpha_color2); //vec4( mix(base_color, cube_color, reflect_factor), 1.0);
	
	///////////////
	//refractCoeff = 1.0003 / refractCoeff;

	vec3 refractVec = normalize(refract(viewVec, normalVec, refractCoeff));
	//refractVec = vec3(0.0,0.0,-1.0);
	
	float b = dot( (positionW.xyz - worldSphereCenter.xyz), refractVec ) * 2.0;
	float a = pow(length(refractVec), 2.0);
	float c = 0.0 - pow(sphereRadius,2.0);
	
	float solution = (-b+sqrt(pow(b,2.0) - 4.0 * a * c))/2.0*a;
	
	vec3 projPoint = (refractVec * (solution-1.0)) + positionW.xyz;
	//projPoint = vec3(positionW.xy, positionW.z - 2*sphereRadius);
		
	float distanceFromCenterToProjPoint = length(projPoint - lightCenterProj);
	
	float lightRatio = smoothstep(0.0, lightProjRadius*0.3 ,lightProjRadius-distanceFromCenterToProjPoint);
	
	//gl_FragColor = vec4(vec3(distanceFromCenterToProjPoint,distanceFromCenterToProjPoint,distanceFromCenterToProjPoint), 1.0);
	//if (solution > 5.5 && solution < 6.5)
	//	gl_FragColor = vec4(1.0,0.0,0.0,1.0);
	//gl_FragColor = vec4(projPoint, 1.0);
	gl_FragColor = vec4(vec3(lightRatio,lightRatio,lightRatio), 1.0);
	//gl_FragColor = vec4(((color.xyz*alpha_color2)+cube_color) * (lightRatio),alpha_color2) ;

}
