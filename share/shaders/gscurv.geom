#version 120 
#extension GL_EXT_gpu_shader4 : enable
#extension GL_EXT_geometry_shader4 : enable

uniform mat4	m2wMatrix;				// world to model matrix.
uniform mat4	pvmMatrix;				// model view projection matrix.
uniform int		uf_dirIndex;			// ray diretion index;

vec4 wPosition[6];
int idx, idy, idz;

varying vec2 segmentParam[2];
varying vec2 segmentEnd[2];
varying float vertKd;

vec3 computeTriangleNormal(int v0, int v1, int v2)
{
	return normalize(cross(wPosition[v1].xyz-wPosition[v0].xyz, wPosition[v2].xyz-wPosition[v1].xyz));	
}

vec3 computeTriangleBBoxSize(int v0, int v1, int v2)
{
	vec3  tmax, tmin;
	tmax = max( max( wPosition[v1].xyz, wPosition[v2].xyz ), wPosition[v0].xyz );
	tmin = min( min( wPosition[v1].xyz, wPosition[v2].xyz ), wPosition[v0].xyz );
	
	return tmax - tmin;
}

bool isZero(in float data)
{
	return abs(data)<1e-4;
}

int computeKdSign(in int i, in vec3 triNorm)
{
	if ( !isZero(triNorm[idz]) )
		return 0;

	// index of the vertex, v3 is the vertex oposite to edge(v1, v2).
	int v1 = i;
	int v2 = (i+2)%6;
	int v3 = (i+4)%6;												

	// v3 is the highest vertex
	if( (wPosition[v3][idz] >= wPosition[v2][idz]) && (wPosition[v3][idz] >= wPosition[v1][idz]) )
		return -1;

	// make v1 to be the higherst vertext
	if( (wPosition[v2][idz] >= wPosition[v1][idz]) )
	{
		int tmpId = v1;
		v1 = v2;
		v2 = tmpId;
	}

	vec3 edgeA = normalize(wPosition[v2].xyz - wPosition[v1].xyz);
	vec3 edgeB = normalize(wPosition[v3].xyz - wPosition[v1].xyz);

	vec3 rayDir = vec3(0.0, 0.0, 0.0);
	rayDir[idz] = -1.0f;

	float tmp = dot(cross(edgeA, rayDir), cross(edgeB, rayDir));
	if ( tmp <= 0.0 )
		return 1;

	if ( dot(edgeA, rayDir) > dot(edgeB, rayDir) )
		return -1;
	else
		return 1;
}

void computeKeKz(int _idx, int _idy, int _idz, vec3 edgeProj, vec3 norm1, vec3 norm2, out vec2 segmentParam)
{
	float ke, kz;
	ke = kz = 0.0;

	ke = edgeProj[_idx]/edgeProj[_idy];

	// testing whether surface is parrel with the ray direction
	if( !isZero(norm1[_idz]) )
		kz = sign(norm1[_idz])*abs(norm1[_idx]/norm1[_idz]);
		
	if( !isZero(norm2[_idz]) )
		kz += sign(norm2[_idz])*abs(norm2[_idx]/norm2[_idz]);

	kz = abs(kz);

	segmentParam  = vec2(kz,  kz*ke*ke);	
}

void main() 
{
	int i=0;
	for(i=0; i<6; i++)
		wPosition[i] = m2wMatrix * gl_PositionIn[i];

	vec3 vTriNorm = computeTriangleNormal(0, 2, 4);
	vec3 vNeigbTriNorm[3];
	vNeigbTriNorm[0] = computeTriangleNormal(0, 1, 2);
	vNeigbTriNorm[1] = computeTriangleNormal(2, 3, 4);
	vNeigbTriNorm[2] = computeTriangleNormal(4, 5, 0);
	
	vec3 vTriBBox = computeTriangleBBoxSize(0, 2, 4);
	vec3 vNeigbTriBBox[3];
	vNeigbTriBBox[0] = computeTriangleBBoxSize(0, 1, 2);
	vNeigbTriBBox[1] = computeTriangleBBoxSize(2, 3, 4);
	vNeigbTriBBox[2] = computeTriangleBBoxSize(4, 5, 0);

	idx = (uf_dirIndex+1)%3;
	idy = (uf_dirIndex+2)%3;
	idz = uf_dirIndex;
		
	// the edge triangles.
	for(i=0; i< gl_VerticesIn; i=i+2)
	{
		segmentEnd[0] = vec2(wPosition[i][idx],			wPosition[i][idy]);
		segmentEnd[1] = vec2(wPosition[(i+2)%6][idx],	wPosition[(i+2)%6][idy]);

		////////////////////Parameter Calculation//////////////////////
		// edge between two adjacent surface
		vec3 edge = cross(vTriNorm, vNeigbTriNorm[i/2]);
		
		// two surface fall on same plane
		if(length(edge)<0.0001)
		{
			segmentParam[0] = vec2(0.0f, 0.0f); 
			segmentParam[1] = vec2(0.0f, 0.0f); 
		}
		else
		{
			// projected edge on XY plane
			vec3 edgePro = edge;
			edgePro[idz] = 0;
			edgePro = normalize(edgePro);

			// edge is paralle with the ray
			if( isZero(edgePro[idx]) && isZero(edgePro[idy]) )
			{
				segmentParam[0] = vec2(0.0f, 0.0f); 
				segmentParam[1] = vec2(0.0f, 0.0f); 
			}
			else
			{
				// edge hit the top and bottom boundary, fill the result in segmentParam[0]
				if( abs(edgePro[idx]) < abs(edgePro[idy]) )
				{
					computeKeKz(idx, idy, idz, edgePro, vTriNorm, vNeigbTriNorm[i/2], segmentParam[0]);
					segmentParam[1] = vec2(0.0f, 0.0f);
				}
				// edge hit the left and right boundary, fill the result in segmentParam[1]
				else
				{
					computeKeKz(idy, idx, idz, edgePro, vTriNorm, vNeigbTriNorm[i/2], segmentParam[1]);
					segmentParam[0] = vec2(0.0f, 0.0f);
				}
			}
		}
		
		
		int curEdgeSign = computeKdSign( i, vTriNorm);
		
		vertKd = wPosition[i][idz]*curEdgeSign;
		gl_Position = pvmMatrix * gl_PositionIn[i];
		EmitVertex();

		vertKd = wPosition[(i+2)%6][idz]*curEdgeSign;
		gl_Position = pvmMatrix * gl_PositionIn[(i+2)%6];
		EmitVertex();

		EndPrimitive();
		
		//// draw the line for the second time, because of the 'half-open' rasterization
		//// for the end points of the segment.
		vertKd = wPosition[(i+2)%6][idz]*curEdgeSign;
		gl_Position = pvmMatrix * gl_PositionIn[(i+2)%6];
		EmitVertex();

		vertKd = wPosition[i][idz]*curEdgeSign;
		gl_Position = pvmMatrix * gl_PositionIn[i];
		EmitVertex();

		EndPrimitive();
	}
}