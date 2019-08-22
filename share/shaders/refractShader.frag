varying vec3 normalVec;
varying vec3 viewVec;

uniform sampler2D sphereTexture;

varying vec3 positionW;
uniform vec3 worldSphereCenter;
uniform vec3 lightCenterProj;
//uniform vec3 discCenter;

uniform float sphereRadius;
uniform float refractCoeff;
uniform float ambientCoeff;

uniform float lightProjRadius;


vec3 reflect(vec3 I, vec3 N)
{
  return I - 2.0 * N * dot(N, I);
}

vec3 refract2(vec3 I, vec3 N, float etaRatio)
{
  float cosI = dot(-I, N);
  float cosT2 = 1.0 - etaRatio * etaRatio *
    (1.0 / (cosI * cosI));
  if (cosT2 > 0.0)
  return  etaRatio * I +
             ((etaRatio * cosI - sqrt(cosT2)) * N);
  else
  	return vec3(0.0,0.0,0.0);
  
}


void main()
{
	///////////////
	//refractCoeff = 1.0003 / refractCoeff;

  vec3 refractVec = normalize(viewVec); //normalize(refract(normalize(viewVec), normalize(normalVec), refractCoeff));
	//refractVec = vec3(0.0,0.0,-1.0);
	
  // length(positionW + t * refractVec - Center)^2 = R^2
  // (px-cx + t*rx)^2 + (py-cy + t*ry)^2 + (pz-cz + t*rz)^2 - R^2 = 0
  // dot(r,r)*t^2 + 2dot(p-c,r)*t + (dot(p-c,p-c)-R^2) = 0
  vec3 dp = (positionW.xyz - worldSphereCenter.xyz);
	float a = dot(refractVec, refractVec);
	float b = dot(dp, refractVec ) * 2.0;
	float c = dot(dp, dp) - sphereRadius*sphereRadius;
	
	float solution = (-b+sqrt(b*b - 4.0 * a * c))/(2.0*a);
	
	vec3 projPoint = (refractVec * (solution)) + positionW.xyz;
	//projPoint = vec3(positionW.xy, positionW.z - 2*sphereRadius);
		
	float distanceFromCenterToProjPoint = length(projPoint - lightCenterProj);
	
	float lightRatio = ambientCoeff+(1.0-ambientCoeff)*smoothstep(0.0, lightProjRadius*0.7 ,lightProjRadius-distanceFromCenterToProjPoint);
	vec4 color = texture2D(sphereTexture,normalize(projPoint-worldSphereCenter.xyz).xy*0.3+vec2(0.5,0.5));
	
	//gl_FragColor = vec4(vec3(distanceFromCenterToProjPoint,distanceFromCenterToProjPoint,distanceFromCenterToProjPoint), 1.0);
	//if (solution > 5.5 && solution < 6.5)
	//	gl_FragColor = vec4(1.0,0.0,0.0,1.0);
	//gl_FragColor = vec4(projPoint, 1.0);
	color.rgb *= lightRatio;
	gl_FragColor = color;
	//gl_FragColor = vec4(((color.xyz*alpha_color2)+cube_color) * (lightRatio),alpha_color2) ;

}
