#version 120
#extension GL_EXT_gpu_shader4 : enable

//varying vec4 diffuse, ambient, /*ambientGlobal,*/ specular;
varying vec3 positionW, normalW;

#ifdef TEXTURE_UNIT_0
uniform sampler2D colorTexture;
#endif //TEXTURE_UNIT_0

#if defined(PLANE_ENVIRONMENT_MAPPING) || defined(BORDER_OPACIFYING) || defined(BORDER_OPACIFYING_V2)
varying vec3 viewVectorW;
#endif

#if defined(BORDER_OPACIFYING) || defined(BORDER_OPACIFYING_V2)
uniform float border_gamma, border_alpha;
#endif

#ifdef PLANE_ENVIRONMENT_MAPPING
uniform sampler2D planeTexture;
uniform float altitude;
#endif //PLANE_ENVIRONMENT_MAPPING

#if defined(PHONG) || defined(BUMP_MAPPING)
varying vec3 normalView;
#endif

#if defined(SMOOTH_LAYER)
uniform float smoothSpecular;
uniform float smoothShininess;
#endif

#if defined(PHONG)
varying vec3 lightDir;
varying float dist;
varying vec3 halfVector;
#endif //PHONG

#if defined(PHONG2)
varying vec3 lightDir2;
varying vec3 halfVector2;
varying float dist2;
#endif //PHONG2


#if defined(BORDER_OPACIFYING_V2) 
varying vec3 lightDirWorld;
uniform float coeffAngle;
#endif

#if defined(TRI_TEXTURING) || defined(BUMP_MAPPING)
uniform vec2 scaleTexture;
#endif

#if defined(TRI_TEXTURING)
varying vec3 restPositionW, restNormalW;
uniform float showDebug;
#if defined(TRI_TEXTURING_SINGLE_TEXTURE)
uniform sampler2D planarTexture;
#define planarTextureX planarTexture
#define planarTextureY planarTexture
#define planarTextureZ planarTexture
#else
uniform sampler2D planarTextureX, planarTextureY, planarTextureZ;
#endif
#endif

#if defined(BUMP_MAPPING)
uniform sampler2D normalMap;
uniform float bumpFactor;
#endif

#ifdef LIGHT2
//varying vec4 diffuse2, specular2;
varying vec3 lightDir2;
#endif //LIGHT2

#if defined (PERLIN_NOISE_COLOR)
uniform float perlinColorFrequency;
uniform int perlinColorOctave;
uniform float perlinColorPersistance;
uniform vec4 perlinColorFactor;
#endif

#if defined (PERLIN_NOISE_BUMP) 
uniform float perlinBumpFrequency;
uniform int perlinBumpOctave;
uniform float perlinBumpPersistance;
uniform float perlinBumpFactor;
#endif

#if defined (PERLIN_NOISE_COLOR) || defined (PERLIN_NOISE_BUMP) 
uniform sampler2D perlinPermutationsTexture;
#endif //PERLIN_NOISE

vec3 reflect(vec3 I, vec3 N)
{
  return I - 2.0 * N * dot(N, I);
}

#if defined (PERLIN_NOISE_COLOR) || defined (PERLIN_NOISE_BUMP) 

int getPerlinValue(int i)
{
	return int(texture2D(perlinPermutationsTexture, vec2((i & 255)/256.0,0))*256);
}

float fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); } // 6t5-15t4+10t3
float dfade(float t) { return 30.0 *t * t * (t*(t - 2.0) + 1.0); } // 30t4-60t3+30t2

float lerp(float t, float a, float b) { return a + t * (b - a); }
float dlerp(float t, float da, float db) { return da + t * (db - da); }
float dlerp(float t, float a, float b, float dt, float da, float db) { return da + t * (db - da) + dt * (b - a); }

float grad(int hash, float x, float y, float z) 
{
      int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
      float u = h<8 ? x : y,                 // INTO 12 GRADIENT DIRECTIONS.
             v = h<4 ? y : h==12||h==14 ? x : z;
      return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
   }

vec4 dgrad(int hash, float x, float y, float z) 
{
      int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
      vec4 u = h<8 ? vec4(1.0, 0.0, 0.0, x) : vec4(0.0, 1.0, 0.0, y),                 // INTO 12 GRADIENT DIRECTIONS.
             v = h<4 ? vec4(0.0, 1.0, 0.0, y) : h==12||h==14 ? vec4(1.0, 0.0, 0.0, x) : vec4(0.0, 0.0, 1.0, z);
      return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}

float noise(float x, float y, float z) 
{
	int X = int(floor(x));                  // FIND UNIT CUBE THAT
	int Y = int(floor(y));                  // CONTAINS POINT.
	int Z = int(floor(z));
	
	x -= floor(x);                                // FIND RELATIVE X,Y,Z
	y -= floor(y);                                // OF POINT IN CUBE.
	z -= floor(z);
	
	float u = fade(x);                                // COMPUTE FADE CURVES
	float v = fade(y);                                // FOR EACH OF X,Y,Z.
	float w = fade(z);
	
    int A  = (getPerlinValue(X)  +Y);
    int AA = (getPerlinValue(A)  +Z);
    int AB = (getPerlinValue(A+1)+Z);
    int B  = (getPerlinValue(X+1)+Y); 
    int BA = (getPerlinValue(B)  +Z);
    int BB = (getPerlinValue(B+1)+Z);      // THE 8 CUBE CORNERS,

    return lerp(w, lerp(v, lerp(u, grad(getPerlinValue(AA), x  , y  , z   ),  // AND ADD
                                   grad(getPerlinValue(BA), x-1, y  , z   )), // BLENDED
                           lerp(u, grad(getPerlinValue(AB), x  , y-1, z   ),  // RESULTS
                                   grad(getPerlinValue(BB), x-1, y-1, z   ))),// FROM  8
                   lerp(v, lerp(u, grad(getPerlinValue(AA+1), x  , y  , z-1 ),  // CORNERS
                                   grad(getPerlinValue(BA+1), x-1, y  , z-1 )), // OF CUBE
                           lerp(u, grad(getPerlinValue(AB+1), x  , y-1, z-1 ),
                                   grad(getPerlinValue(BB+1), x-1, y-1, z-1 ))));
}

vec4 dnoise(float x, float y, float z) 
{
	int X = int(floor(x));                  // FIND UNIT CUBE THAT
	int Y = int(floor(y));                  // CONTAINS POINT.
	int Z = int(floor(z));
	
	x -= floor(x);                                // FIND RELATIVE X,Y,Z
	y -= floor(y);                                // OF POINT IN CUBE.
	z -= floor(z);
	
	float u = fade(x);                                // COMPUTE FADE CURVES
	float v = fade(y);                                // FOR EACH OF X,Y,Z.
	float w = fade(z);
	
    int A  = (getPerlinValue(X)  +Y);
    int AA = (getPerlinValue(A)  +Z);
    int AB = (getPerlinValue(A+1)+Z);
    int B  = (getPerlinValue(X+1)+Y); 
    int BA = (getPerlinValue(B)  +Z);
    int BB = (getPerlinValue(B+1)+Z);      // THE 8 CUBE CORNERS,
    vec4 g0 = dgrad(getPerlinValue(AA)  , x  , y  , z   );
    vec4 g1 = dgrad(getPerlinValue(BA)  , x-1, y  , z   );
    vec4 g2 = dgrad(getPerlinValue(AB)  , x  , y-1, z   );
    vec4 g3 = dgrad(getPerlinValue(BB)  , x-1, y-1, z   );
    vec4 g4 = dgrad(getPerlinValue(AA+1), x  , y  , z-1 );
    vec4 g5 = dgrad(getPerlinValue(BA+1), x-1, y  , z-1 );
    vec4 g6 = dgrad(getPerlinValue(AB+1), x  , y-1, z-1 );
    vec4 g7 = dgrad(getPerlinValue(BB+1), x-1, y-1, z-1 );
    vec4 res;
    res.w = lerp(w, lerp(v, lerp(u,g0.w,  // AND ADD
                                   g1.w), // BLENDED
                           lerp(u, g2.w,  // RESULTS
                                   g3.w)),// FROM  8
                   lerp(v, lerp(u, g4.w,  // CORNERS
                                   g5.w), // OF CUBE
                           lerp(u, g6.w,
                                   g7.w)));
    float d = 0.01;
    float u2 = u+d*dfade(x);
    float v2 = v+d*dfade(y);
    float w2 = w+d*dfade(z);

    res.x = lerp(w, lerp(v, lerp(u2,g0.w+g0.x*d,  // AND ADD
                                    g1.w+g1.x*d), // BLENDED
                           lerp(u2, g2.w+g2.x*d,  // RESULTS
                                    g3.w+g3.x*d)),// FROM  8
                   lerp(v, lerp(u2, g4.w+g4.x*d,  // CORNERS
                                    g5.w+g5.x*d), // OF CUBE
                           lerp(u2, g6.w+g6.x*d,
                                    g7.w+g7.x*d))) - res.w;

    res.y = lerp(w, lerp(v2, lerp(u,g0.w+g0.y*d,  // AND ADD
                                    g1.w+g1.y*d), // BLENDED
                            lerp(u, g2.w+g2.y*d,  // RESULTS
                                    g3.w+g3.y*d)),// FROM  8
                   lerp(v2, lerp(u, g4.w+g4.y*d,  // CORNERS
                                    g5.w+g5.y*d), // OF CUBE
                            lerp(u, g6.w+g6.y*d,
                                    g7.w+g7.y*d))) - res.w;

    res.z = lerp(w2, lerp(v, lerp(u,g0.w+g0.z*d,  // AND ADD
                                    g1.w+g1.z*d), // BLENDED
                            lerp(u, g2.w+g2.z*d,  // RESULTS
                                    g3.w+g3.z*d)),// FROM  8
                    lerp(v, lerp(u, g4.w+g4.z*d,  // CORNERS
                                    g5.w+g5.z*d), // OF CUBE
                            lerp(u, g6.w+g6.z*d,
                                    g7.w+g7.z*d))) - res.w;
                                    
    res.xyz *= 100.0;
    return res;
}

float noise(vec3 p)
{
	return noise(p.x,p.y,p.z);
}
vec4 dnoise(vec3 p)
{
	return dnoise(p.x,p.y,p.z);
}

float perlin_noise(vec3 p, float freq, int octave, float persistance)
{
    float res = 0.0;
    float o=freq;
    float f=persistance;
        
    for(int i = 0;i<octave;i++)
    {
        res += noise(p*o) * f;
    	o *= 2.0;
    	f *= f;
    }
           
    return res;
}

vec4 perlin_dnoise(vec3 p, float freq, int octave, float persistance)
{
    vec4 res = vec4(0.0,0.0,0.0,0.0);
    float o=freq;
    float f=persistance;
        
    for(int i = 0;i<octave;i++)
    {
        res += dnoise(p*o) * f;
    	o *= 2.0;
    	f *= f;
    }
           
    return res;
}

#endif //PERLIN_NOISE

void main()
{
	vec4 color = gl_Color;
	
	color = gl_LightModel.ambient * gl_FrontMaterial.ambient;

	//normal
	vec3 n;
	
#ifdef TEXTURE_UNIT_0
	color.rgb = texture2D(colorTexture,gl_TexCoord[0].st).rgb;
#endif //TEXTURE_UNIT_0

#ifdef TRI_TEXTURING
	// To compute unlit color and normal, only vPositionW and vNormalW should be used, so that it is independant of the current deformation
	//XY = 1 0 0
	//XZ = 0 1 0
	//YZ = 0 0 1
	color = vec4(0.0, 0.0, 0.0, 0.0);
	vec3 pos0 = (restPositionW);
	//vec3 n0 = normalize(vNormalW);
	vec3 n0 = normalize(restNormalW);

	vec3 coefs = abs(n0)-vec3(0.2,0.2,0.2);
	coefs *= 7.0;
	coefs = coefs*coefs*coefs ;//pow(coefs,3.0);
	coefs = max(vec3(0.0,0.0,0.0),coefs);
	coefs /= dot(coefs,vec3(1.0,1.0,1.0)); // make sum = 1

	//XY -> Z
	color += texture2D(planarTextureZ,vec2(pos0.x/scaleTexture.x,pos0.y/scaleTexture.y) ) * coefs.z;
	//XZ -> Y
	color += texture2D(planarTextureY,vec2(pos0.x/scaleTexture.x,pos0.z/scaleTexture.y) ) * coefs.y;
	//YZ -> X
	color += texture2D(planarTextureX,vec2(pos0.y/scaleTexture.x,pos0.z/scaleTexture.y) ) * coefs.x;

	color.rgb = showDebug * (vec3(1.0,1.0,1.0)-coefs) + (1.0-showDebug)*color.rgb;
	color.a = gl_FrontMaterial.diffuse.a;
#endif

#if defined (PERLIN_NOISE_COLOR) 
	color.a = 1.0;
	//color *= 0.8+0.2*noise(positionW*10);
	//color *= 0.8+0.2*perlin_noise(positionW, 4, 1.0);
	
	//color = perlinColorFactor + (perlin_noise(positionW, perlinColorFrequency, perlinColorOctave, perlinColorPersistance));
	
	float perlinColor = perlin_noise(positionW, perlinColorFrequency, perlinColorOctave, perlinColorPersistance);
	color.rgb =  perlinColorFactor.rgb * (color.rgb+perlinColor);
	//color.a += (perlinColor);
	
	//color *= vec4(0.0,0.0,0.0,0.0)*(perlin_noise(positionW, perlinColorFrequency, perlinColorOctave, perlinColorPersistance));
	//color = perlinColorFactor;
	//float t = perlin_noise(positionW, 4, 1.0);
	//color.xyz = 0.8+vec3(t, t, t);
#endif

#if defined(BUMP_MAPPING) && defined(TRI_TEXTURING)
	vec3 bumpCurrentNormal =  n0;
	vec3 bumpCurrentPosition =  pos0;
	vec3 bumpCoefs =  coefs;
	vec4 bump = vec4(0.0,0.0,0.0,0.0);// = normalize( texture2D(normalMap, vec2(pos.x/scaleTexture.x,pos.y/scaleTexture.y) ).xyz * 2.0 - 1.0);
	// Compute bump normal
	vec3 bx = vec3(0.0,bumpCurrentNormal.z,-bumpCurrentNormal.y); 
	vec3 by = vec3(-bumpCurrentNormal.z,0.0,bumpCurrentNormal.x); 
	vec3 bz = vec3(bumpCurrentNormal.y,-bumpCurrentNormal.x,0.0); 
	//XY -> Z
	//color.g = pos0.y;
	bump.yxw += texture2D(normalMap,vec2(bumpCurrentPosition.x/scaleTexture.x,bumpCurrentPosition.y/scaleTexture.y) ).xyz * bumpCoefs.z;
	//XZ -> Y
	//color.r = pos0.x;
	bump.zxw += texture2D(normalMap,vec2(bumpCurrentPosition.x/scaleTexture.x,bumpCurrentPosition.z/scaleTexture.y) ).xyz * bumpCoefs.y;
	//YZ -> X
	//color.b = pos0.z;
	bump.zyw += texture2D(normalMap,vec2(bumpCurrentPosition.y/scaleTexture.x,bumpCurrentPosition.z/scaleTexture.y) ).xyz * bumpCoefs.x;

	bump *= bumpFactor;
	bump.w += 1.0-bumpFactor;
	
	n = normalize(gl_NormalMatrix*( normalize(normalW) * bump.w + bx * bump.x + by * bump.y + bz * bump.z));
#endif //defined(BUMP_MAPPING) && defined(TRI_TEXTURING)
	
#if defined(BUMP_MAPPING) && !defined(TRI_TEXTURING)
	n = normalize(texture2D(normalMap, gl_TexCoord[0].xy ).xyz * 2.0 - 1.0);
#endif //BUMP_MAPPING

#if defined(PHONG) && !defined(BUMP_MAPPING)
	//normal as usual
	n = normalize(normalView);
#endif //PHONG

#if defined (PERLIN_NOISE_BUMP) 
	//n += gl_NormalMatrix*dnoise(positionW*10).xyz*0.2;
	n += gl_NormalMatrix*perlin_dnoise(positionW,perlinBumpFrequency,perlinBumpOctave,perlinBumpPersistance).xyz*perlinBumpFactor;
	n = normalize(n);
#endif //PERLIN_NOISE_BUMP
	

#ifdef BORDER_OPACIFYING
	color = gl_LightModel.ambient * gl_FrontMaterial.ambient;
	
	vec3 unitNormalVec = normalize(normalW);
	vec3 unitViewVec = normalize(viewVectorW);
	
	color.a = color.a + (border_alpha - color.a)* (pow( 1.0 - abs(dot(unitNormalVec,unitViewVec)), border_gamma));

	color.rgb *= color.a;
	
#endif

#ifdef BORDER_OPACIFYING_V2
	color = gl_LightModel.ambient * gl_FrontMaterial.ambient;
	vec3 unitNormalVec = normalize(normalW);
	vec3 unitViewVec = normalize(viewVectorW);
	 
	float dotAngle = abs(dot(unitNormalVec,lightDirWorld));
	dotAngle *= coeffAngle;
	
	color.a = color.a + (border_alpha - color.a)* (pow( 1.0 - abs(dot(unitNormalVec,unitViewVec)), border_gamma));
	
	color.rgb = color.rgb* (dotAngle);
	
#endif


#ifdef PHONG
	float NdotL,NdotHV;
	float att,spotEffect;
	vec4 phong_color = vec4(0.0,0.0,0.0,1.0);
	
	/* compute the dot product between normal and ldir */
#ifdef DOUBLE_SIDED
	NdotL = abs(dot(n,normalize(lightDir)));
#else
	NdotL = max(dot(n,normalize(lightDir)),0.0);
#endif

	//vec3 halfV = normalize(gl_LightSource[0].halfVector.xyz);
	vec3 halfV = halfVector;
	vec4 spec = vec4(0.0,0.0,0.0,0.0);

#ifdef DOUBLE_SIDED
	NdotHV = abs(dot(n,halfV));
#else
	NdotHV = max(dot(n,halfV),0.0);
#endif
	
	if (NdotL > 0.0)
	{
		float spotEffect = dot(normalize(gl_LightSource[0].spotDirection), normalize(-lightDir));
	
		if (spotEffect > gl_LightSource[0].spotCosCutoff)
		{
			spotEffect = smoothstep(gl_LightSource[0].spotCosCutoff, 1.0, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
			att = spotEffect /* / (gl_LightSource[0].constantAttenuation +
					gl_LightSource[0].linearAttenuation * dist +
					gl_LightSource[0].quadraticAttenuation * dist * dist) */;
	
			//phong_color += (diffuse * NdotL) /* * att */;
			phong_color.rgb += (gl_LightSource[0].diffuse.rgb * NdotL) * att ;
	
        	spec =  gl_LightSource[0].specular * pow(NdotHV,gl_FrontMaterial.shininess) * att;

#if defined(SMOOTH_LAYER)
            spec +=  smoothSpecular * pow( max(dot(normalize(normalW),halfV),0.0),smoothShininess) * att;
#endif //SMOOTH_LAYER

		}
	}


#ifdef PHONG2
	NdotL = max(dot(n,normalize(lightDir2)),0.0);
	
	if (NdotL > 0.0)
	{
/*
		float spotEffect = dot(normalize(gl_LightSource[1].spotDirection), normalize(-lightDir2));
	
		if (spotEffect > gl_LightSource[1].spotCosCutoff)
		{
			spotEffect = smoothstep(gl_LightSource[1].spotCosCutoff, 1.0, spotEffect); //pow(spotEffect, gl_LightSource[0].spotExponent);
			att = spotEffect / (gl_LightSource[1].constantAttenuation +
					gl_LightSource[1].linearAttenuation * dist2 +
					gl_LightSource[1].quadraticAttenuation * dist2 * dist2);
	*/
			//phong_color += (diffuse * NdotL) /* * att */;
			phong_color.rgb += (gl_LightSource[1].diffuse.rgb * NdotL);//  * att ;


#ifdef PHONG2
       	vec3 halfV2 = halfVector2;
	    NdotHV = max(dot(n,halfV2),0.0);
	    spec +=  gl_LightSource[1].specular * pow(NdotHV,gl_FrontMaterial.shininess) /* * att */;
#endif

/*
		}
*/
	}
#endif
	
#if defined(TEXTURE_UNIT_0) || defined(TRI_TEXTURING)
	color.rgb *= gl_FrontMaterial.diffuse.rgb*phong_color.rgb;
#else
	color.rgb += gl_FrontMaterial.diffuse.rgb*phong_color.rgb;
#endif //defined(TEXTURE_UNIT_0) || defined(TRI_TEXTURING)
	
	color += gl_FrontMaterial.specular*spec;
	
#endif //PHONG

#ifdef PLANE_ENVIRONMENT_MAPPING
	color = gl_LightModel.ambient * gl_FrontMaterial.ambient;
	
	vec3 reflectVec = reflect(viewVectorW, normalW);
	
	//if ((reflectVec.z)>0.0)
	  color.rgb += texture2D(planeTexture, reflectVec.xy*( altitude/reflectVec.z )+vec2(0.5,0.5)).rgb * gl_FrontMaterial.specular.rgb ;

	
#endif //PLANE_ENVIRONMENT_MAPPING

	// Write the final pixel.
	gl_FragColor = color;
	//gl_FragColor = vec4(1.0,1.0,1.0,1.0);
	//vec4 p = texture(perlinPermutationsTexture,gl_TexCoord[0].st);

}
