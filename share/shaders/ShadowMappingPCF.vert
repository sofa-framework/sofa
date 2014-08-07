
void main()
{
	
	vec4 vertex;	
   
    vertex = gl_ModelViewMatrix * gl_Vertex;
   
    gl_TexCoord[1] = gl_TextureMatrix[1] * vertex;
 
    gl_Position = ftransform();
   
}