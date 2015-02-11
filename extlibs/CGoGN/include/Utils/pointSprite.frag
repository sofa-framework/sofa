// PointSprite::fragmentShaderText

uniform mat4 ProjectionMatrix;
uniform vec3 lightPos;
uniform vec3 ambiant;
uniform float size;

#ifdef WITH_PLANE
uniform vec3 eyePos;
VARYING_FRAG vec3 shiftedEye;
#endif

#ifdef WITH_COLOR_PER_VERTEX 
VARYING_FRAG vec4 colorsprite;
#else
uniform vec4 colorsprite;
#endif

VARYING_FRAG vec2 spriteCoord;
VARYING_FRAG vec3 sphereCenter;
FRAG_OUT_DEF ;

void main(void)
{
#ifdef WITH_PLANE
	vec3 billboard_frag_pos = vec3(spriteCoord, 0.0) * size;
	vec3 ray_direction = normalize(billboard_frag_pos - shiftedEye);
	float av = dot(shiftedEye,ray_direction);
	float arg = av*av - dot(shiftedEye,shiftedEye) + size*size;
	if (arg< 0.0)
		discard;
	float t = -av - sqrt(arg);
	vec3 frag_position_eye = ray_direction * t + eyePos	;
#else
	vec3 billboard_frag_pos = sphereCenter + vec3(spriteCoord, 0.0) * size;
	vec3 ray_direction = normalize(billboard_frag_pos);
	float TD = -dot(ray_direction,sphereCenter);
	float c = dot(sphereCenter, sphereCenter) - size * size;
	float arg = TD * TD - c;
    
	if (arg < 0.0)
		discard;

	float t = -c / (TD - sqrt(arg));
	vec3 frag_position_eye = ray_direction * t ;
#endif	

	vec4 pos = ProjectionMatrix * vec4(frag_position_eye, 1.0);
	gl_FragDepth = (pos.z / pos.w + 1.0) / 2.0;

	vec3 N = normalize(frag_position_eye - sphereCenter);
	vec3 L = normalize (lightPos - frag_position_eye);
	float lambertTerm = dot(N,L);
	
	vec4 result = colorsprite*lambertTerm;
	result.xyz += ambiant;

	FRAG_OUT = result;
}
