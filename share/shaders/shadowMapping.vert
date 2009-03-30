
varying vec3 normal;

varying vec4 ambientGlobal;
#ifdef SHADOW_LIGHT0
varying vec4 shadowTexCoord0;
varying vec4 diffuse;
varying vec3 lightDir,halfVector;
varying float dist;
#endif

#ifdef SHADOW_LIGHT1
varying vec4 shadowTexCoord1;
varying vec4 diffuse1;
varying vec3 lightDir1,halfVector1;
varying float dist1;
#endif

void main()
{
	vec4 ecPos;
	vec3 aux;

	/* first transform the normal into eye space and normalize the result */
	normal = normalize(gl_NormalMatrix * gl_Normal);

	/* now normalize the light's direction. Note that according to the
	OpenGL specification, the light is stored in eye space.*/
	ecPos = gl_ModelViewMatrix * gl_Vertex;

	ambientGlobal = gl_LightModel.ambient * gl_FrontMaterial.ambient;

#ifdef SHADOW_LIGHT0

	aux = vec3(gl_LightSource[0].position-ecPos);
	lightDir = /*normalize*/(aux);

	/* compute the distance to the light source to a varying variable*/
	dist = length(aux);

	/* Normalize the halfVector to pass it to the fragment shader */
	halfVector = normalize(gl_LightSource[0].halfVector.xyz);

	/* Compute the diffuse, ambient and globalAmbient terms */
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
	ambientGlobal += gl_FrontMaterial.ambient * gl_LightSource[0].ambient;

	shadowTexCoord0 = gl_TextureMatrix[0] * gl_ModelViewMatrix * gl_Vertex;
#endif

	///////////////////
#ifdef SHADOW_LIGHT1

	/* now normalize the light's direction. Note that according to the
	OpenGL specification, the light is stored in eye space.*/
	aux = vec3(gl_LightSource[1].position-ecPos);
	lightDir1 = /*normalize*/(aux);

	/* compute the distance to the light source to a varying variable*/
	dist1 = length(aux);

	/* Normalize the halfVector to pass it to the fragment shader */
	halfVector1 = normalize(gl_LightSource[1].halfVector.xyz);

	/* Compute the diffuse, ambient and globalAmbient terms */
	diffuse1 = gl_FrontMaterial.diffuse * gl_LightSource[1].diffuse;
	ambientGlobal += gl_FrontMaterial.ambient * gl_LightSource[1].ambient;

	shadowTexCoord1 = gl_TextureMatrix[1] * gl_ModelViewMatrix * gl_Vertex;
#endif

	//////
	gl_Position = ftransform();
	gl_FrontColor = gl_FrontMaterial.diffuse;
	gl_BackColor = gl_BackMaterial.diffuse;
}
