#version 330 core

in vec3 VertexPosition;

uniform mat4 ModelViewProjectionMatrix;

invariant gl_Position;

void main ()
{
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}

