#version 120

//0 -> disabled, 1 -> only lighting, 2 -> lighting & shadow
uniform int u_shadowTextureUnit[MAX_NUMBER_OF_LIGHTS];
uniform int u_lightFlags[MAX_NUMBER_OF_LIGHTS];
uniform int u_lightTypes[MAX_NUMBER_OF_LIGHTS];
uniform vec3 u_lightDirs[MAX_NUMBER_OF_LIGHTS];
uniform mat4 u_lightProjectionMatrices[MAX_NUMBER_OF_LIGHTS];
uniform mat4 u_lightModelViewMatrices[MAX_NUMBER_OF_LIGHTS];

varying vec3 normal;
varying vec4 ambientGlobal;
varying vec3 lightDirs[MAX_NUMBER_OF_LIGHTS];
varying vec4 lightSpacePosition[MAX_NUMBER_OF_LIGHTS];

#if ENABLE_SHADOW == 1 
varying vec4 shadowTexCoord[MAX_NUMBER_OF_LIGHTS];
//uniform float shadowNormalBias;
#endif // ENABLE_SHADOW == 1 
void main()
{
	vec4 ecPos;
	vec3 aux;

	// first transform the normal into eye space and normalize the result
	normal = normalize(gl_NormalMatrix * gl_Normal);

	// now normalize the light's direction. Note that according to the
	//OpenGL specification, the light is stored in eye space.
	ecPos = gl_ModelViewMatrix * gl_Vertex;

#if ENABLE_SHADOW == 1
//       vec4 shadowPos = ecPos + vec4(normal * 1.0 /* shadowNormalBias */, 0.0);
#endif

	ambientGlobal = gl_LightModel.ambient * gl_FrontMaterial.ambient;

	for (int i=0 ; i<MAX_NUMBER_OF_LIGHTS ;i++)
	{
		if (u_lightFlags[i] > 0)
		{
			lightSpacePosition[i] = u_lightProjectionMatrices[i] * u_lightModelViewMatrices[i] * gl_Vertex;

			if(u_lightTypes[i] == 0)
				lightDirs[i] = (gl_ModelViewMatrix * vec4((u_lightDirs[i] * -1), 0.0)).xyz;
			else if(u_lightTypes[i] == 2)
			{
				lightDirs[i] = (gl_LightSource[i].position-ecPos).xyz;
				//spotOff[i] = gl_LightSource[i].spotCosCutoff;
			}


#if ENABLE_SHADOW == 1 
			if (u_lightFlags[i] == 2)
            {
                shadowTexCoord[i] = gl_TextureMatrix[u_shadowTextureUnit[i]] * gl_ModelViewMatrix * gl_Vertex;

            }
#endif // ENABLE_SHADOW == 1 
		}

	}

#ifdef USE_TEXTURE
	gl_TexCoord[0] = gl_MultiTexCoord0;
#endif

	//////
	gl_Position = ftransform();
	gl_FrontColor = gl_FrontMaterial.diffuse;
	gl_BackColor = gl_BackMaterial.diffuse;
}