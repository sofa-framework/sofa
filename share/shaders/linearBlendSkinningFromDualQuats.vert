/*************************************************************
GLSL - Linear Blend Skinning using dual quaternions.
Used when dual quaternions are available while the skinning weights are designed for linear blend.
A more efficient implementation may be possible, avoiding the computation of matrices.
(c) 2015 Armelle BAUER, IMAGINE(INRIA)-GMCAO(TIMC)
*************************************************************/

// Dual quaternion skinning inputs
#define MAX_DUALQUATS_SIZE 64			// OpenGL 3.x ensures at least 1024/4 = 256 uniform vec4. Hint: use an uniform buffer object for more elements.
uniform vec4 dualQuats[MAX_DUALQUATS_SIZE];

attribute vec4 indices;
attribute vec4 weights;

// Light
varying vec3 normal;
varying vec3 view;

//function to convert quat to matrices...
mat4 toMatrix( int index )
{
	int baseOffset = index * 2;
	vec4 orientation = dualQuats[baseOffset].xyzw;
	vec4 dual = dualQuats[baseOffset+1].xyzw;
	
	float m_00 = (1.0 - 2.0 * (orientation[1] * orientation[1] + orientation[2] * orientation[2]));
	float m_01 = (2.0 * (orientation[0] * orientation[1] - orientation[2] * orientation[3]));
	float m_02 = (2.0 * (orientation[2] * orientation[0] + orientation[1] * orientation[3]));
	float m_03 = 2.0 * ( -dual[3]*orientation[0] + dual[0]*orientation[3] - dual[1]*orientation[2] + dual[2]*orientation[1] );
	
	float m_10 = (2.0 * (orientation[0] * orientation[1] + orientation[2] * orientation[3]));
	float m_11 = (1.0 - 2.0 * (orientation[2] * orientation[2] + orientation[0] * orientation[0]));
	float m_12 = (2.0 * (orientation[1] * orientation[2] - orientation[0] * orientation[3]));
	float m_13 = 2.0 * ( -dual[3]*orientation[1] + dual[0]*orientation[2] + dual[1]*orientation[3] - dual[2]*orientation[0] );
	
	float m_20 = (2.0 * (orientation[2] * orientation[0] - orientation[1] * orientation[3]));
	float m_21 = (2.0 * (orientation[1] * orientation[2] + orientation[0] * orientation[3]));
	float m_22 = (1.0 - 2.0 * (orientation[1] * orientation[1] + orientation[0] * orientation[0]));
	float m_23 = 2.0 * ( -dual[3]*orientation[2] - dual[0]*orientation[1] + dual[1]*orientation[0] + dual[2]*orientation[3] );
	
	float m_30 = 0.0;
	float m_31 = 0.0;
	float m_32 = 0.0;
	float m_33 = 1.0;
	
	//vec4 column0 = vec4(m_00, m_01, m_02, m_03);
	//vec4 column1 = vec4(m_10, m_11, m_12, m_13);
	//vec4 column2 = vec4(m_20, m_21, m_22, m_23);
	//vec4 column3 = vec4(m_30, m_31, m_32, m_33);
	vec4 column0 = vec4(m_00, m_10, m_20, m_30);
	vec4 column1 = vec4(m_01, m_11, m_21, m_31);
	vec4 column2 = vec4(m_02, m_12, m_22, m_32);
	vec4 column3 = vec4(m_03, m_13, m_23, m_33);
	
	mat4 quatToMat =  mat4(column0, column1, column2, column3);//= mat2x4( dualQuats[baseOffset].wxyz, dualQuats[baseOffset+1].wxyz );
	
	return quatToMat;
}

void main()
{
    vec4 pos = toMatrix(int(indices[0])) * gl_Vertex * weights[0];
	pos += toMatrix(int(indices[1])) * gl_Vertex * weights[1];
	pos += toMatrix(int(indices[2])) * gl_Vertex * weights[2];
	pos += toMatrix(int(indices[3])) * gl_Vertex * weights[3];
	
	gl_Position = gl_ModelViewProjectionMatrix * pos;
	
	//-------------------------------------------------------
	
	vec4 normal4 = vec4(gl_Normal.xyz, 0.0);
        vec4 norm = toMatrix(int(indices[0])) * normal4 * weights[0];
	norm += toMatrix(int(indices[1])) * normal4 * weights[1];
	norm += toMatrix(int(indices[2])) * normal4 * weights[2];
	norm += toMatrix(int(indices[3])) * normal4 * weights[3];
	
	view = vec3(gl_ModelViewMatrix * pos);
	normal = gl_NormalMatrix*norm.xyz;
}
