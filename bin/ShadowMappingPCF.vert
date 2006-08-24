varying vec3 position;
varying vec4 vertex;
varying vec3 normal;

void main()
{
    position = gl_Vertex.xyz;

    vertex = gl_ModelViewMatrix * gl_Vertex;
    normal = gl_NormalMatrix    * gl_Normal;

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_TexCoord[1] = gl_TextureMatrix[1] * vertex;
   

  // globalAmbient = gl_LightModel.ambient * gl_FrontMaterial.ambient;
   
  // gl_FrontColor = gl_FrontMaterial.ambient;
   	vec3 v3Normal;		
		
	float fAngle;
	float fShininessFactor;
	
	// transform the vertext normal the same the object is transformed
	v3Normal   = gl_NormalMatrix * gl_Normal;

	// set normal length to 1.
	v3Normal   = normalize(v3Normal);

	// calculate the angle eye-position - vertex - light direction
	// the angle must not be less than 0.0
	fAngle = max(0.0, dot(v3Normal, vec3(gl_LightSource[0].halfVector)));
	
	// calculate the vertex shininess as the calculated angle powered to the materials shininess factor
	fShininessFactor = pow(fAngle, gl_FrontMaterial.shininess);
	
	// calculate the vertex color with the normal gouraud lighting calculation
	gl_FrontColor = gl_LightSource[0].ambient * gl_FrontMaterial.ambient +
       			    gl_LightSource[0].diffuse * gl_FrontMaterial.diffuse * fAngle +
		            gl_LightSource[0].specular * gl_FrontMaterial.specular * fShininessFactor;

    gl_Position = ftransform();
}