#version 120

//0 -> disabled, 1 -> only lighting, 2 -> lighting & shadow
uniform int lightFlag[MAX_NUMBER_OF_LIGHTS];
varying vec3 normal;
varying vec4 ambientGlobal;
varying vec3 lightDir[MAX_NUMBER_OF_LIGHTS];
//varying float dist[MAX_NUMBER_OF_LIGHTS];
//varying float spotOff[MAX_NUMBER_OF_LIGHTS];

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
		if (lightFlag[i] > 0)
		{
			aux = vec3(gl_LightSource[i].position-ecPos);
			lightDir[i] = (aux);

			// compute the distance to the light source to a varying variable
			//dist[i] = length(aux);

			// Normalize the halfVector to pass it to the fragment shader
			//halfVector[i] = normalize(gl_LightSource[i].halfVector.xyz);

			// Compute the diffuse, ambient and globalAmbient terms
			//diffuse[i] = gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse;
			//ambientGlobal += gl_FrontMaterial.ambient * gl_LightSource[i].ambient;
			//spotOff[i] = gl_LightSource[i].spotCosCutoff;

#if ENABLE_SHADOW == 1 
			if (lightFlag[i] == 2)
            {
				//shadowTexCoord[i] = shadowMatrix[i] * shadowPos;
                float NormalOffsetScale = -1.0; //0.2 /* * max(1.0, 1.0 - 0.5*dot(normal,normalize(aux))) */ * pow(abs(dot(gl_LightSource[i].spotDirection,-aux)),0.5);
                vec3 ShadowOffset = normal * NormalOffsetScale;
                shadowTexCoord[i] = gl_TextureMatrix[i] * (ecPos + vec4(ShadowOffset, 0.0));
                //shadowTexCoord[i] = gl_TextureMatrix[i] * ecPos;
                //shadowTexCoord[i].x += dot(gl_TextureMatrix[i][0].xyz, ShadowOffset);
                //shadowTexCoord[i].y += dot(gl_TextureMatrix[i][1].xyz, ShadowOffset);
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
