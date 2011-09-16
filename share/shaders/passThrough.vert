
void main() 
{
	gl_Position = gl_Vertex;// gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_FrontColor = gl_Color;
}
