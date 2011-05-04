#version   120
#extension GL_EXT_geometry_shader4: enable


// TO the FRAGMENT SHADER
varying out vec2 dataFragment;

// FROM the VERTEX SHADER
varying in  float field[4];

// UNIFORMS: arrays to perform:
//	- the mapping on the basis graph 
//	- create the resulting triangles
uniform vec4  MappingTable[32];
uniform vec4  RunSelectTable[10];
//	- can be used to modulated depth0 if necessary (not used here)
//	- see the generation of the primitives for more details
uniform float modulation;

uniform vec3 center;
uniform	float rmin;
uniform	float rmax;
uniform	float dmin;
uniform	float dmax;

// Constants
const vec4 zero = vec4(0);

/***********************************************************************
	AUXILIARY FUNCTIONS: INSPIRED FROM Wylie et al. VERTEX SHADER
************************************************************************/

// Returns the values in z of the cross product  v0 x v1 and v0 x v2
vec2 zcross(in vec4 u0, in vec4 u1, in vec4 u2){

	vec4 tmp = vec4(vec2(u0.xy*u1.yx),vec2(u0.xy*u2.yx));
	return vec2(tmp.x - tmp.y, tmp.z - tmp.w);
}

// Make the computations of the tests 1, 2 and 4 of Wylie et al.
// Return a vec4 = (t1, t2, t4, ?)
// t1 : diff1 is between diff2 and diff3
// t2 : diff2 is between diff1 and diff3
// t4 : diff1-> diff2 are in the counterclockwise order
 bvec4 test124(in vec4 u0, in vec4 u1, in vec4 u2, in vec4 u3){
 	
 	bvec4 tests;
 	
 	vec4 diff1 = u1-u0;
 	vec4 diff2 = u2-u0;
 	vec4 diff3 = u3-u0;
 	
 	vec4 zcrosses;
 	
  	zcrosses.xy = zcross(diff1,diff2,diff3);
 	zcrosses.zw = zcross(diff3,diff2,vec4(1.0/diff3.y,0,0,0));
 	// zcrosses.w = -1 to make test 4 : zcrosses.x > 0 in the same instruction.
 	
 	tests.xyz   = lessThan((zcrosses.xxx*zcrosses.yzw), zero.xyz);
	tests.w     = false;
 	
 	return tests;
 	
 }

 // Return the mapping of the vertex according to the mask
 vec4 multiplex(in vec4 mask, in vec4 x0, in vec4 x1, in vec4 x2, in vec4 x3){
 	
 	vec4 result;
 	
 	result = mask.x*x0;
 	result = mask.y*x1 + result;
 	result = mask.z*x2 + result;
 	result = mask.w*x3 + result;
 	
 	return result;
 }
 	
 // Project the vertices on the basis graph
 void mapVertices(ivec4 tests, 
 				  in  vec4 v0in,  in  vec4 v1in,  in  vec4 v2in,  in  vec4 v3in,
 				  out vec4 v0out, out vec4 v1out, out vec4 v2out, out vec4 v3out,
 				  in  vec4 scalin, out vec4 scalout) 
 {
 	int index;
 	vec4 mask;
 	
 	index   = 4*(tests.z + 2*tests.y + 4*tests.x);
 	
 	v0out   = multiplex(MappingTable[index], v0in, v1in, v2in, v3in);
 	scalout.x = dot(MappingTable[index],scalin);
 	
 	v1out   = multiplex(MappingTable[index+1], v0in, v1in, v2in, v3in);
 	scalout.y = dot(MappingTable[index+1],scalin);
 	
  	v2out   = multiplex(MappingTable[index+2], v0in, v1in, v2in, v3in);
 	scalout.z = dot(MappingTable[index+2],scalin);
 	
  	v3out   = multiplex(MappingTable[index+3], v0in, v1in, v2in, v3in);
 	scalout.w = dot(MappingTable[index+3],scalin);

 }
 
 // Compute the intersection point between v0v1 and v2v3 in the plane xy-
 // i1 is the intersection on v0v1 and i2 on v2v3
 // Only the z coordinates is considered different
 // interp.x = dist(v0, i1)/dist(v0,v1) and interp.y = dist(v2,i2)/dist(v2,v3)
 // interp.z is the denominator (used only in this procedure)
 void compute_intersection(in vec4 u0, in vec4 u1, in vec4 u2, in vec4 u3, out vec4 interp, out vec4 i1, out vec4 i2){
 	
 	vec4  A, B, C;
 	
 	A = u1-u0;
 	B = u3-u2;
 	C = u0-u2;
 
 	interp.xyz = vec3(B.x*C.y - B.y*C.x, A.x*C.y - A.y*C.x, A.x*B.y - A.y*B.x);
 	interp.xy  = interp.xy/interp.z;
 	
 	i1 = u0 + interp.x*A;
 	i2 = u2 + interp.y*B;
}

// Depth of the current vertex (used only for the vertex 0 of the basis graph)
float compute_depth(in vec4 interp, in vec4 i1, in vec4 i2, in bool test3) {
	/*
	 	float depth_i;
	 	float depth_v1;
	 	float depth;
	 	
	 	depth_i  = abs(i1.z-i2.z);
	 	
	 	depth_v1 = depth_i/interp.x;
	 	
	 	depth = test3 ? depth_i : depth_v1;
	 	
 		return depth_i;
	*/
	//float depth_i = abs(i1.z-i2.z);
	float depth_i = abs(i1.w-i2.w);
	if (interp.x > 1.0)
	   return depth_i / interp.x;
	else if (interp.x < 0.0)
	   return depth_i / (1-interp.x);
	else
           return depth_i;
}

// Compute the position of the vertex 0 of the basis graph
vec4 compute_positionZero(in bool test3, 
		      in vec4 u0, in vec4 u1, in vec4 u2, in vec4 u3,
		      in vec4 i1, in vec4 i2)
{

	int index;    
	vec4 position; 
	
	index    = int(test3);
	
	position = multiplex(RunSelectTable[index], u0, u1, u2, u3);
	return mix(position, (i1+i2)*0.5, float(test3));
		
}

// Compute the position of the other vertices of the basis graph
vec4 compute_positionnonZero(in bool test3, 
		      in vec4 u0, in vec4 u1, in vec4 u2, in vec4 u3,
		      in int number)
{

	int index;   
	vec4 position; 
	
	index    = 2*number+int(test3);
	
	position = multiplex(RunSelectTable[index], u0, u1, u2, u3);
	return position;
		
}	 

//  Compute the scalar field of the vertex 0 of the basis graph
float compute_scalarfieldZero(in bvec4 tests, in vec4 scal, in vec4 interp) {

	vec2 is;     //scalar field of i1 (is.x) and i2 (is.y)

	if (tests.w) {
		is.xy = vec2(scal.x + interp.x*(scal.y-scal.x),scal.z + interp.y*(scal.w-scal.z));
	} else {
		is.xy   = vec2(scal.y, scal.x + (scal.z + interp.y*(scal.w-scal.z) - scal.x)/interp.x); 
	} 

	
	return  0.5*(is.x+is.y);
	
}

// Compute the scalar field of the other vertices of the basis graph
float compute_scalarfieldnonZero(in bvec4 tests, in vec4 scal, in int number, in vec4 interp) {

	int index;   
	vec4 tmp;
	float scalend;
		
	index    = 2*number+int(tests.w);
	
	tmp     = RunSelectTable[index];

	scalend = dot(tmp,scal);

	return scalend;
	
}

// Discarding degenerated triangle
//	return true if not degenerated, false otherwise
bool notDegeneratedTriangle(in vec4 u, in vec4 v, in vec4 w)
{

	return (!(all(equal(u, v)) || all(equal(u, w)) || all(equal(v, w))));
}

float computeGradualOpacity(in vec3 v)
{
float dist = length(center - v);
if (dist < rmin)
{
return dmax;
}
if (dist < rmax)
{
dist -= rmin;
float ratio = rmax - rmin;
float d = dmax - dmin;
return dmin + (d*pow(pow(1-dist/ratio,2),2)*(4*dist/ratio+1));
}

return	dmin;
}

/***********************************************************************
	MAIN: COMPUTATIONS OF THE TRIANGLES
************************************************************************/

void main(void){

//var 	

vec4  u0, u1, u2, u3; //current point
vec4  w0, w1, w2, w3; //projected into the basis graph
vec4  sfield;	      //original scalar field 
vec4  newSField;      //scalar field associated to the basis graph
bvec4 tests;          //test to determine the projection
vec4  interp;         //intepolation of the middle point
vec4  i1, i2;         //middle of two segments

//outputs	

float depth0;
float scalField[5];
vec4  position[5];

//

u0 = gl_PositionIn[0]; //u0.xyz *= 1.0/u0.w; u0.w = 1.0;
u1 = gl_PositionIn[1]; //u1.xyz *= 1.0/u1.w; u1.w = 1.0;
u2 = gl_PositionIn[2]; //u2.xyz *= 1.0/u2.w; u2.w = 1.0;
u3 = gl_PositionIn[3]; //u3.xyz *= 1.0/u3.w; u3.w = 1.0;

// Compute the tests 1,2 and 4 to determine the graph basis (according to Wylie et al.)
tests = test124(u0, u1, u2, u3);

sfield = vec4(field[0], field[1], field[2], field[3]);

// Mapping the vertices onto the graph basis
mapVertices(ivec4(tests), u0, u1, u2, u3, w0, w1, w2, w3, sfield, newSField);

// Computing the intersection to find the vertex 0
compute_intersection(w0, w1, w2, w3, interp, i1, i2);

//Last test (test 3 according to Wylie et al.) 
tests.w = bool(interp.x < 1.);

depth0       = compute_depth(interp, i1, i2, tests.w);
scalField[0] = compute_scalarfieldZero(tests, newSField, interp);
position[0]  = compute_positionZero(tests.w, w0, w1, w2, w3, i1, i2);
position[0].w = 1.0;

for(int i=1;i<5;i++){
scalField[i] = compute_scalarfieldnonZero(tests, newSField, i, interp);
position[i]  = compute_positionnonZero(tests.w, w0, w1, w2, w3, i);
position[i].w = 1.0;
}

if (notDegeneratedTriangle(position[0], position[1], position[2])){
//TRIANGLE 0
dataFragment = vec2(scalField[1], 0);
gl_Position = position[1];
EmitVertex();

dataFragment = vec2(scalField[2], 0);
gl_Position = position[2];
EmitVertex();

//depth0 can need to be modulate with a constant > 10 to have a stronger opacity (uniform modulation)
//depends of the size of the cell!
dataFragment = vec2(scalField[0], depth0*computeGradualOpacity(position[0].xyz));
gl_Position = position[0];
EmitVertex();

EndPrimitive();
}

if (notDegeneratedTriangle(position[0], position[2], position[3]))
{
//TRIANGLE 1
dataFragment = vec2(scalField[2], 0);
gl_Position  = position[2];
EmitVertex();

dataFragment = vec2(scalField[3], 0);
gl_Position  = position[3];
EmitVertex();

//depth0 can need to be modulate with a constant > 10 to have a stronger opacity (uniform modulation)
//depends of the size of the cell and of the number of cells!
dataFragment = vec2(scalField[0], depth0*computeGradualOpacity(position[0].xyz));
gl_Position  = position[0];
EmitVertex();

EndPrimitive();
}

if (notDegeneratedTriangle(position[0], position[3], position[4]))
{
//TRIANGLE 2

//depth0 can need to be modulate with a constant > 10 to have a stronger opacity (uniform modulation)
//depends of the size of the cell and of the number of cells!
dataFragment = vec2(scalField[0], depth0*computeGradualOpacity(position[0].xyz));
gl_Position = position[0];
EmitVertex();

dataFragment = vec2(scalField[3], 0);
gl_Position = position[3];
EmitVertex();

dataFragment = vec2(scalField[4], 0);
gl_Position = position[4];
EmitVertex();

EndPrimitive();
}

if (notDegeneratedTriangle(position[0], position[1], position[4]))
{
//TRIANGLE 3
dataFragment = vec2(scalField[4], 0);
gl_Position = position[4];
EmitVertex();

dataFragment = vec2(scalField[1], 0);
gl_Position = position[1];
EmitVertex();

//depth0 can need to be modulate with a constant > 10 to have a stronger opacity (uniform modulation)
//depends of the size of the cell and of the number of cells!
dataFragment = vec2(scalField[0], depth0*computeGradualOpacity(position[0].xyz));
gl_Position = position[0];
EmitVertex();

//EndPrimitive() sous-entendu
}

}