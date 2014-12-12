#version 330 compatibility

/* Dual Quaternion Skinning

  from http://www.seas.upenn.edu/~ladislav/dq/index.html

  Version 1.0.3, November 1st, 2007

  Copyright (C) 2006-2007 University of Dublin, Trinity College, All Rights 
  Reserved

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the author(s) be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Author: Ladislav Kavan, ladislav.kavan@gmail.com
*/

// and Per pixel Phong lighting for one spot light

// Dual quaternion skinning inputs
#define MAX_DUALQUATS_SIZE 64			// OpenGL 3.x ensures at least 1024/4 = 256 uniform vec4. Hint: use an uniform buffer object for more elements.
uniform vec4 dualQuats[MAX_DUALQUATS_SIZE];

in vec4 indices;
in vec4 weights;

// Phong
out vec3 normal;
out vec3 view;

mat2x4 getBoneDQ( int index )
{
	int baseOffset = index * 2;
	mat2x4 boneDQ = mat2x4( dualQuats[baseOffset].wxyz, dualQuats[baseOffset+1].wxyz );
	return boneDQ;
}


void main()
{
	// (I)nitial position and normal
	vec3 positionI = gl_Vertex.xyz;
	vec3 normalI = gl_Normal.xyz;

	// Blending dual quat
	mat2x4 blendDQ = weights[0] * getBoneDQ(int(indices[0]));
	blendDQ += weights[1] * getBoneDQ(int(indices[1]));
	blendDQ += weights[2] * getBoneDQ(int(indices[2]));
	blendDQ += weights[3] * getBoneDQ(int(indices[3]));

	float len = length(blendDQ[0]);
	blendDQ /= len;

	// Computing new position
	vec3 dqPosition = positionI + 2.0*cross(blendDQ[0].yzw, cross(blendDQ[0].yzw, positionI) + blendDQ[0].x*positionI);
	vec3 translation = 2.0*(blendDQ[0].x*blendDQ[1].yzw - blendDQ[1].x*blendDQ[0].yzw + cross(blendDQ[0].yzw, blendDQ[1].yzw));
	dqPosition += translation;

	// Computing new normal
	vec3 dqNormal = normalI + 2.0*cross(blendDQ[0].yzw, cross(blendDQ[0].yzw, normalI) + blendDQ[0].x*normalI);

	// Computing output of the vertex shader
	vec4 dqPosition4 = vec4(dqPosition, 1.0);

	gl_Position = gl_ModelViewProjectionMatrix * dqPosition4;
	normal = gl_NormalMatrix * dqNormal;
	view = vec3(gl_ModelViewMatrix * dqPosition4);
}
