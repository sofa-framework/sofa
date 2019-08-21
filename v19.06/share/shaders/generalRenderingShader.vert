
//General Settings () 
//varying vec4 diffuse, ambient, /*ambientGlobal,*/ specular;
varying vec3 positionW, normalW;
 
#if defined(PLANE_ENVIRONMENT_MAPPING) || defined(BORDER_OPACIFYING)  || defined(BORDER_OPACIFYING_V2) 
varying vec3 viewVectorW;
#endif //PLANE_ENVIRONMENT_MAPPING

#if defined(PHONG) || defined(BUMP_MAPPING)
varying vec3 normalView;
#endif

#if defined(PHONG)
varying vec3 lightDir;
varying vec3 halfVector;
varying float dist;
#endif //PHONG

#if defined(PHONG2)
varying vec3 lightDir2;
varying vec3 halfVector2;
varying float dist2;
#endif //PHONG2

#if defined(BORDER_OPACIFYING_V2) 
varying vec3 lightDirWorld;
#endif

#if defined(TRI_TEXTURING)
varying vec3 restPositionW, restNormalW;
attribute vec3 restPosition, restNormal;
#endif

#ifdef LIGHT2
varying vec4 diffuse2, specular2;
varying vec3 lightDir2;
#endif //LIGHT2

void main()
{
	gl_Position = ftransform();
	
	gl_FrontColor = gl_FrontMaterial.diffuse;
	gl_BackColor = gl_BackMaterial.diffuse;
	
	positionW = gl_Vertex.xyz;
	normalW = normalize(gl_Normal);
	
	
	/* Compute the diffuse, ambient and globalAmbient terms */
//	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
//	ambient = (gl_LightModel.ambient + gl_LightSource[0].ambient) * gl_FrontMaterial.ambient;
	//ambientGlobal = gl_LightModel.ambient * gl_FrontMaterial.ambient;
	
#ifdef WET_SPECULAR
	//specular = vec4(1.0,1.0,1.0,1.0);
#else
	//specular = gl_FrontMaterial.specular * gl_LightSource[0].specular;
#endif

#ifdef TEXTURE_UNIT_0
	gl_TexCoord[0] = gl_MultiTexCoord0;
	//gl_TexCoord[1] = gl_MultiTexCoord1;
#endif //TEXTURE_UNIT_0

#if defined(TRI_TEXTURING)
	restPositionW = restPosition;
	restNormalW = restNormal;
#endif
	
#if defined(PLANE_ENVIRONMENT_MAPPING) || defined(BORDER_OPACIFYING) || defined(BORDER_OPACIFYING_V2) 
	vec3 eyePositionW = gl_ModelViewMatrixInverse[3].xyz;
	viewVectorW = positionW - eyePositionW;

#endif
		
#if defined(PHONG) || defined(BUMP_MAPPING)
	normalView = gl_NormalMatrix * gl_Normal;
#endif

#if defined(PHONG)
	vec4 ecPos = gl_ModelViewMatrix * gl_Vertex;
	vec3 aux = vec3(gl_LightSource[0].position-ecPos);
	
	//aux = (gl_ModelViewMatrixInverse*gl_LightSource[0].position).xyz - gl_Vertex.xyz;
	lightDir = normalize(aux);
	dist = length(aux);
	halfVector = vec3(gl_LightSource[0].halfVector);
#endif //PHONG

#if defined(BORDER_OPACIFYING_V2) 
	//lightDirWorld =  (gl_ModelViewMatrixInverse*gl_LightSource[0].position).xyz - gl_Vertex.xyz;
	lightDirWorld =  normalize(gl_LightSource[0].position.xyz - gl_Vertex.xyz);
#endif

#if defined(PHONG2)
	vec3 aux2 = vec3(gl_LightSource[1].position - ecPos);
	
	//aux = (gl_ModelViewMatrixInverse*gl_LightSource[0].position).xyz - gl_Vertex.xyz;
	lightDir2 = normalize(aux2);
	dist2 = length(aux2);
	halfVector2 = vec3(gl_LightSource[1].halfVector);
	
#endif //PHONG2


}
