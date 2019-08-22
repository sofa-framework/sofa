varying vec3 normalVec;
varying vec3 viewVec;

varying vec3 positionW;

void main()
{
	
	gl_TexCoord[0] = gl_MultiTexCoord0;
	//gl_TexCoord[0] = gl_MultiTexCoord1;	
	
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	positionW = gl_Vertex.xyz;
	
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
	//normalView = gl_NormalMatrix * gl_Normal;
	
	//sphereCenter = (gl_ModelViewProjectionMatrix * vec4(worldSphereCenter,1.0)).xyz;
	//lightCenter = (gl_ModelViewProjectionMatrix * vec4(lightCenterProj,1.0)).xyz;
	
	
}
