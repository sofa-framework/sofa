#version 330

// Precision qualifier (le meme que dans le vert. ou highp par defaut !)
precision highp float;

// Entrees
smooth in vec3 EyeVector, Normal, LightDir;

// Uniforms (material)
uniform vec4 materialDiffuse;
uniform vec4 materialSpecular;
uniform vec4 materialAmbient;
uniform float shininess;


// Sortie
out vec4 FragColor;

void main()
{
	vec3 N = normalize (Normal);
	vec3 L = normalize (LightDir);
	
	float lambertTerm = dot(N,L);
	
	vec4 Color = materialAmbient;
	
	if(lambertTerm > 0.0)
	{
		Color += materialDiffuse * lambertTerm;	
		
		vec3 E = normalize(EyeVector);
		vec3 R = reflect(-L, N);
		float specular = pow( max(dot(R, E), 0.0), shininess );
		Color += materialSpecular * specular;	
	}

	FragColor = Color;
}

