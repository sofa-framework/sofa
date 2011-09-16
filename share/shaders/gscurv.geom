#version 150 
#extension GL_EXT_geometry_shader4 : enable

uniform mat4	w2mMatrix;			// world to model matrix.
uniform mat4	pvmMatrix;			// model view projection matrix.

float point_segment_distance(vec3 v, vec3 e1, vec3 e2)
{
	vec3	e = normalize( e1-e2 );
	float prj = dot((v-e1), e);

	float dist = sqrt( distance(v, e1) * distance(v, e1) - prj * prj);

	return dist;
}

vec3 computeCurvature(int i)
{
	int v1 = i;
	int v2 = (i+2)%6;
	int vo = (i+1)%6;		// index of the out face vertex.
	int vi = (i+4)%6;		// index of the inner face vertex.
	
	vec3 edge = gl_PositionIn[v2].xyz - gl_PositionIn[v1].xyz;
	vec3 eo = gl_PositionIn[vo].xyz - gl_PositionIn[v1].xyz;
	vec3 ei = gl_PositionIn[vi].xyz - gl_PositionIn[v1].xyz;
	
	vec3 no = normalize(cross(edge, eo));
	vec3 ni = normalize(cross(ei, edge));
	float fTemp = clamp(dot(no, ni), -1.0, 1.0);
	float beta = acos(fTemp);

	vec3  ne = normalize((no + ni) * 0.5);		// normal of the edge
	vec3  e  = normalize(edge);					// direction of the edge.
	vec3  crossNEE = normalize(cross(ne, e));

	mat3 curvMat = mat3(ne, crossNEE, e);
	mat3 drMat = mat3( 0.5*beta, 0.0,	   0.0,
					   0.0,      0.5*beta, 0.0,
					   0.0,      0.0,      0.0);

	// get the curvature matrix of current edge.
	curvMat = curvMat * drMat * transpose(curvMat);

	// compute curvature for each ray direction.
	mat3	revMat = mat3x3(w2mMatrix);

	// transform ray directions from world coord to model local coord.
	vec3 r0 = revMat * vec3(1,0,0);
	vec3 r1 = revMat * vec3(0,1,0);
	vec3 r2 = revMat * vec3(0,0,1);

	vec3 curv = vec3( abs(dot(curvMat*r0, r0)), 
					  abs(dot(curvMat*r1, r1)),
					  abs(dot(curvMat*r2, r2)));
	curv = vec3( atan(curv.x)/1.57, atan(curv.y)/1.57, atan(curv.z)/1.57);
			
	return curv;
}

void main() 
{
	int i=0;
	
	// the edge triangles.
	for(i=0; i< gl_VerticesIn; i=i+2)
	{
		// compute edge curvature
		vec3 curv = computeCurvature(i);

		gl_Position = pvmMatrix * gl_PositionIn[i];
		gl_FrontColor = vec4(curv, 1.0); // gl_FrontColorIn[i]; //
		EmitVertex();

		gl_Position = pvmMatrix * gl_PositionIn[(i+2)%6];
		gl_FrontColor = vec4(curv, 1.0); // gl_FrontColorIn[i]; //
		EmitVertex();

		gl_Position = pvmMatrix * gl_PositionIn[(i+2)%6];
		gl_FrontColor = vec4(curv, 1.0); // gl_FrontColorIn[i]; //
		EmitVertex();

		EndPrimitive();
	}
}