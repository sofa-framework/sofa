varying vec3 normalVec;
varying vec3 viewVec;

varying vec4 diffuse, ambient, specular;
varying vec3 lightDir, /* halfVector, */ normalView;
//varying float dist;

void main()
{
	gl_TexCoord[0] = gl_MultiTexCoord0;
	//gl_TexCoord[0] = gl_MultiTexCoord1;	 
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_FrontColor = gl_FrontMaterial.diffuse;
	gl_BackColor = gl_BackMaterial.diffuse;
	// Compute position and normal in world space
	vec3 positionW = gl_Vertex.xyz; ;
	vec3 normalW = normalize(gl_Normal);
	
	vec3 eyePositionW = gl_ModelViewMatrixInverse[3].xyz;
	
	// Compute the incident and reflected vectors
	vec3 I = positionW - eyePositionW;
	//reflectVec = reflect(I, normalW);
	normalVec = normalW;
	viewVec = I;
	normalView = gl_NormalMatrix * gl_Normal;
	vec4 ecPos = gl_ModelViewMatrix * gl_Vertex;
	vec3 aux = vec3(gl_LightSource[0].position-ecPos);
	lightDir = normalize(aux);
	
	//dist = length(aux);
	
	//halfVector = normalize(gl_LightSource[0].halfVector.xyz);
	
	/* Compute the diffuse, ambient and globalAmbient terms */
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
	ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient + gl_LightModel.ambient * gl_FrontMaterial.ambient;
	specular = gl_FrontMaterial.specular * gl_LightSource[0].specular;
	

}
