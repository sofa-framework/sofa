// ShaderCustomTex::vertexShaderText

ATTRIBUTE vec3 VertexPosition, VertexNormal;
ATTRIBUTE vec2 VertexTexCoord;
uniform mat4 TransformationMatrix;
uniform mat4 ModelViewProjectionMatrix;
VARYING_VERT vec2 texCoord;
VARYING_VERT vec3 Normal;
INVARIANT_POS;
void main ()
{
	Normal = vec3 (ModelViewProjectionMatrix * TransformationMatrix * vec4 (VertexNormal , 1.0));
	gl_Position = ModelViewProjectionMatrix * TransformationMatrix * vec4 (VertexPosition, 1.0);
	texCoord = VertexTexCoord;
}

