
#ifdef PLANE_ENVIRONMENT_MAPPING
varying vec3 normalVec;
varying vec3 viewVec;
#endif

#ifdef PHONG
varying vec3 lightDir, /* halfVector, */ normalView;
#endif

varying vec4 diffuse, ambient, specular;

void main()
{
	gl_Position = ftransform();
	
	gl_FrontColor = gl_FrontMaterial.diffuse;
	gl_BackColor = gl_BackMaterial.diffuse;

#ifdef TEXTURE_UNIT_0
	gl_TexCoord[0] = gl_MultiTexCoord0;
	//gl_TexCoord[1] = gl_MultiTexCoord1;
#endif
	
#ifdef PLANE_ENVIRONMENT_MAPPING
	// Compute position and normal in world space
	vec3 positionW = gl_Vertex.xyz; ;
	vec3 normalW = normalize(gl_Normal);
	
	vec3 eyePositionW = gl_ModelViewMatrixInverse[3].xyz;
	
	// Compute the incident and reflected vectors
	vec3 I = positionW - eyePositionW;
	//reflectVec = reflect(I, normalW);
	normalVec = normalW;
	viewVec = I;
#endif
		
#ifdef PHONG
	normalView = gl_NormalMatrix * gl_Normal;
	vec4 ecPos = gl_ModelViewMatrix * gl_Vertex;
	vec3 aux = vec3(gl_LightSource[0].position-ecPos);
	lightDir = normalize(aux);
#endif

	/* Compute the diffuse, ambient and globalAmbient terms */
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
	ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient + gl_LightModel.ambient * gl_FrontMaterial.ambient;
	specular = gl_FrontMaterial.specular * gl_LightSource[0].specular;


}
