#version 330
// Attributs
in vec3 VertexPosition, VertexNormal;

// Uniforms
// matrices
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;
uniform mat4 NormalMatrix;
//light position (lumiere blanche)
uniform vec3 LightPosition;

// sortie vers le shader suivant (frag)
smooth out vec3 EyeVector, Normal, LightDir;

// pour invariant (pas vraiment compris ...)
invariant gl_Position;

void main ()
{
	Normal = vec3 (NormalMatrix * vec4 (VertexNormal, 0.0));
	vec3 Position = vec3 (ModelViewMatrix * vec4 (VertexPosition, 1.0));

	LightDir = LightPosition - Position;
	EyeVector = -Position;
	
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}

