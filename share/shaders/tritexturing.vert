varying vec3 normalVec;
varying vec3 viewVec;
varying vec4 vPosition, vPositionW;

varying vec4 diffuse, ambient, ambientGlobal, specular;
varying vec3 lightDir, normalView, halfVector;
varying float dist;

attribute vec3 vTangent;
varying vec3 lightVec; 
varying vec3 eyeVec;
varying vec3 spotDir;

void main()
{
	gl_Position = ftransform();
	gl_FrontColor = gl_FrontMaterial.diffuse;
	gl_BackColor = gl_BackMaterial.diffuse;

	//gl_TexCoord[0] = gl_MultiTexCoord0;
	
	// Compute position and normal in world space
	vec3 positionW = gl_Vertex.xyz; ;
	vec3 normalW = normalize(gl_Normal);
	
	vec3 eyePositionW = gl_ModelViewMatrixInverse[3].xyz;
	
	// Compute the incident and reflected vectors
	vec3 I = positionW - eyePositionW;
	//reflectVec = reflect(I, normalW);
	normalVec = normalW;
	viewVec = I;
	//vPosition = gl_ModelViewMatrix * gl_Vertex;
	vPositionW = gl_Vertex;
	
	vPositionW.x = 0.5 * (1.0 + vPositionW.x);
	vPositionW.y = 0.5 * (1.0 + vPositionW.y);
	vPositionW.z = 0.5 * (1.0 + vPositionW.z);
	vPositionW.w = 1.0;
	
	normalView = gl_NormalMatrix * gl_Normal;
	vec4 ecPos = gl_ModelViewMatrix * gl_Vertex;
	vec3 aux = vec3(gl_LightSource[0].position-ecPos);
	dist = length(aux);
	lightDir = normalize(aux);

	
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
	ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
	ambientGlobal = gl_LightModel.ambient * gl_FrontMaterial.ambient;
	specular = gl_FrontMaterial.specular * gl_LightSource[0].specular;
	
	vec3 n = normalize(gl_NormalMatrix * gl_Normal);
	vec3 t = normalize(gl_NormalMatrix * vTangent);
	vec3 b = normalize(cross(n, t));
	t = cross(b,n);
	
	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);
	vec3 tmpVec = gl_LightSource[0].position.xyz - vVertex;
	
	lightVec.x = dot(tmpVec, t);
	lightVec.y = dot(tmpVec, b);
	lightVec.z = dot(tmpVec, n);
	lightDir = lightVec;
	tmpVec = -vVertex;
	eyeVec.x = dot(tmpVec, t);
	eyeVec.y = dot(tmpVec, b);
	eyeVec.z = dot(tmpVec, n);
	tmpVec = gl_LightSource[0].spotDirection;
	spotDir.x = dot(tmpVec, t);
	spotDir.y = dot(tmpVec, b);
	spotDir.z = dot(tmpVec, n); 
	tmpVec = gl_LightSource[0].halfVector.xyz;
	halfVector.x = dot(tmpVec, t);
	halfVector.y = dot(tmpVec, b);
	halfVector.z = dot(tmpVec, n); 
	halfVector = normalize(halfVector);
}
