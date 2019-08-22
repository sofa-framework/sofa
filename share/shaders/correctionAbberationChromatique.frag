#version 120
//Shader permettant la correction des aberrations chromatiques dues aux lentilles de l'oculus rift.
//Dans le cas du changement des lentilles, il faut ajuster les facteurs des composantes.

uniform sampler2D obj;


uniform float width,height;
void main (void)
{
	vec4 final_color = vec4(1.0,1.0,0.0,1.0);

	float factX= 0.987; //On etire la composante rouge
	float factY= 1.00;
	float factZ= 1.02;	//On compresse la composante bleu

	vec2 st = gl_TexCoord[0].st;
	float xr = (st.x-0.5)*factX +0.5;
	float yr = (st.y-0.5)*factX +0.5;

	float xg = (st.x-0.5)*factY +0.5;
	float yg = (st.y-0.5)*factY +0.5;

	float xb = (st.x-0.5)*factZ +0.5;
	float yb = (st.y-0.5)*factZ +0.5;

	final_color.x = texture2D(obj, vec2(xr,yr)).x;
	final_color.y = texture2D(obj, vec2(xg,yg)).y;
	final_color.z = texture2D(obj, vec2(xb,yb)).z;

	gl_FragColor = final_color;
}
