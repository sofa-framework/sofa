// ShaderCustom::vertexShaderText
uniform mat4 TransformationMatrix;
uniform mat4 ModelViewProjectionMatrix;
ATTRIBUTE vec3 VertexPosition;
void main()
{
	gl_Position = TransformationMatrix *vec4(VertexPosition, 1.0);
}
