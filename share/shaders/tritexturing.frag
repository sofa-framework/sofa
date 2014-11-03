uniform sampler2D planarTextureX, planarTextureY, planarTextureZ;

//varying vec3 normalVec;
//varying vec3 viewVec;

uniform vec2 scaleTexture;
uniform float showDebug;
uniform float bumpFactor;

varying vec4 diffuse, ambient, specular;
varying vec3 lightDir, normalView, halfVector;
varying float dist;

varying vec3 spotDir;

varying vec3 restPositionW, restNormalW;

uniform sampler2D normalMap;

void main()
{
	// To compute unlit color and normal, only vPositionW and vNormalW should be used, so that it is independant of the current deformation
	//XY = 1 0 0
	//XZ = 0 1 0
	//YZ = 0 0 1
	vec4 color = vec4(0.0,0.0,0.0,0.0);//gl_Color;
	vec3 pos0 = (restPositionW);
	//vec3 n0 = normalize(vNormalW);
	vec3 n0 = normalize(restNormalW);

	vec3 coefs = abs(n0)-vec3(0.2,0.2,0.2);
	coefs *= 7.0;
	coefs = coefs*coefs*coefs ;//pow(coefs,3.0);
	coefs = max(vec3(0.0,0.0,0.0),coefs);
	coefs /= dot(coefs,vec3(1.0,1.0,1.0)); // make sum = 1

	// Write the final pixel.
	//gl_FragColor = texture2D(planarTexture,gl_TexCoord[0].st);
	//XY -> Z
	//color.g = pos0.y;
	color += texture2D(planarTextureZ,vec2(pos0.x/scaleTexture.x,pos0.y/scaleTexture.y) ) * coefs.z;
	//XZ -> Y
	//color.r = pos0.x;
	color += texture2D(planarTextureY,vec2(pos0.x/scaleTexture.x,pos0.z/scaleTexture.y) ) * coefs.y;
	//YZ -> X
	//color.b = pos0.z;
	color += texture2D(planarTextureX,vec2(pos0.y/scaleTexture.x,pos0.z/scaleTexture.y) ) * coefs.x;

	color.rgb = showDebug * (vec3(1.0,1.0,1.0)-coefs) + (1.0-showDebug)*color.rgb;

	color.a = 1.0;
	
	vec4 bump = vec4(0.0,0.0,0.0,0.0);// = normalize( texture2D(normalMap, vec2(pos.x/scaleTexture.x,pos.y/scaleTexture.y) ).xyz * 2.0 - 1.0);
	// Compute bump normal
	vec3 bx = vec3(0.0,n0.z,-n0.y); //cross(n0,vec3(1.0,0.0,0.0));
	vec3 by = vec3(-n0.z,0.0,n0.x); //cross(n0,vec3(0.0,1.0,0.0));
	vec3 bz = vec3(n0.y,-n0.x,0.0); //cross(n0,vec3(0.0,0.0,1.0));
	//XY -> Z
	//color.g = pos0.y;
	bump.yxw += texture2D(normalMap,vec2(pos0.x/scaleTexture.x,pos0.y/scaleTexture.y) ).xyz * coefs.z;
	//XZ -> Y
	//color.r = pos0.x;
	bump.zxw += texture2D(normalMap,vec2(pos0.x/scaleTexture.x,pos0.z/scaleTexture.y) ).xyz * coefs.y;
	//YZ -> X
	//color.b = pos0.z;
	bump.zyw += texture2D(normalMap,vec2(pos0.y/scaleTexture.x,pos0.z/scaleTexture.y) ).xyz * coefs.x;

	bump *= bumpFactor;
	bump.w += 1.0-bumpFactor;
	
	//vec3 unitNormalVec = normalize(normalVec);
	vec3 n = normalize(gl_NormalMatrix*( normalize(normalView) * bump.w + bx * bump.x + by * bump.y + bz * bump.z));
	//n = normalize(normalView);
	//vec3 unitViewVec = normalize(viewVec);

	//Phong
	vec3 halfV;
	float NdotL,NdotHV;
	float att,spotEffect;
	vec4 phong_color = color * ambient;
	
	//n = normalize(bumpFactor*bump + vec3(0,0,1.0-bumpFactor));
	
	/* compute the dot product between normal and ldir */
	NdotL = max(dot(n,normalize(lightDir)),0.0);

	if (NdotL > 0.0) {
	
		spotEffect = dot(normalize(spotDir), normalize(-lightDir));
		if (spotEffect > gl_LightSource[0].spotCosCutoff) {
			spotEffect = pow(spotEffect, gl_LightSource[0].spotExponent);
			att = spotEffect / (gl_LightSource[0].constantAttenuation +
					gl_LightSource[0].linearAttenuation * dist +
					gl_LightSource[0].quadraticAttenuation * dist * dist);
				
			phong_color += att *  (diffuse*color * NdotL) ;
			
			halfV = normalize(halfVector);
			NdotHV = max(dot(n,halfV),0.0);
			phong_color += att * specular * pow(NdotHV,gl_FrontMaterial.shininess);
			
		}
	}
	
	gl_FragColor = phong_color;
	//gl_FragColor = vec4(restNormalW,1.0);
}
