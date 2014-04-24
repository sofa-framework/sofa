// ShaderSimpleColor::vertexShaderText

ATTRIBUTE vec3 VertexPosition, VertexNormal;
uniform mat4 ModelViewProjectionMatrix;

void main ()
{
	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);
}
