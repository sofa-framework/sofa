
#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform vec2	uf_vpCenter;
uniform vec2	uf_cellSize;
uniform vec2	uf_bboxCenter;

varying vec2 segmentParam[2];
varying vec2 segmentEnd[2];
varying float vertKd;

void getSegInFragment(out float lx, out float ly, out float end0, out float end1)
{
	end0 = 0.0f;
	end1 = 0.0f;
	
	// compute fragment center position in world coordinate.
	vec2 dCoord = gl_FragCoord.xy - uf_vpCenter;
	vec2 center = uf_bboxCenter + dCoord * uf_cellSize;

	vec2 boundX = vec2(center.x, center.x) + uf_cellSize.x*vec2(-0.5, 0.5);
	vec2 boundY = vec2(center.y, center.y) + uf_cellSize.y*vec2(-0.5, 0.5);

	// compute segment-square intersection.
	// line equation is ax+by+c=0
	float a = segmentEnd[1].y - segmentEnd[0].y;
	float b = segmentEnd[0].x - segmentEnd[1].x;
	float c = segmentEnd[0].y*(segmentEnd[1].x - segmentEnd[0].x) - segmentEnd[0].x*(segmentEnd[1].y - segmentEnd[0].y);

	int count=0;
	float tmpX, tmpY;
	vec2 intr[2];

	// test segment intersection with the left and right edge of the square.
	for(int i=0; i<2; i++)
	{
		tmpX = boundX[i];
		tmpY = -(a*tmpX + c)/b;
		if( (tmpY - boundY.x)>-1e-5 && (tmpY - boundY.y)<1e-5)
		{
			intr[count] = vec2(tmpX, tmpY);
			count += 1;
		}
	}

	// test segment intersection with the top and bottom edge of the square.
	for(int i=0; (i<2) && (count<2); i++)
	{
		tmpY = boundY[i];
		tmpX = -(b*tmpY + c)/a;
		if( (tmpX - boundX.x)>-1e-5 && (tmpX - boundX.y)<1e-5)
		{
			if(count == 1 && distance(intr[0], vec2(tmpX, tmpY)) < 1e-5)
			{	
			}
			else
			{
				intr[count] = vec2(tmpX, tmpY);
				count += 1;
			}
		}
	}

	lx = ly = 0.0f;
	if ( count == 2 )
	{
		vec2 used[2];
		vec2 intrEndLen[2];
		float segLength = distance(segmentEnd[0], segmentEnd[1]);
		intrEndLen[0] = vec2( distance(segmentEnd[0], intr[0]), distance(segmentEnd[1], intr[0]) ) / segLength;
		intrEndLen[1] = vec2( distance(segmentEnd[0], intr[1]), distance(segmentEnd[1], intr[1]) ) / segLength;
		int ucount=0;
		for( int i=0; i<2; i++)
		{
			if (ucount > 1)
				break;

			if( (intrEndLen[i].x<1.0) && (intrEndLen[i].y<1.0) )
			{
				used[ucount] = intr[i];
				ucount++;
			}
			else 
			{
				if( intrEndLen[i].x<intrEndLen[i].y )
				{
					used[ucount] = segmentEnd[0];
					end0 = 1.0f;
					ucount++;
				}
				else
				{
					used[ucount] = segmentEnd[1];
					end1 = 1.0f;
					ucount++;
				}
			}
		}

		if(ucount == 2)
		{
			lx = abs(used[0].x - used[1].x);
			ly = abs(used[0].y - used[1].y);
		}
	}
}

void main() 
{
	float lx, ly;
	float end0, end1;
	getSegInFragment(lx, ly, end0, end1);

	float tmp1 = 1.0/8.0;
	float tmp2 = 1.0/24.0;
	float tmp3 = 1.0/2.0;

	float mnCoef = tmp1*segmentParam[0].x*lx + tmp1*segmentParam[1].x*ly;
	float m2Coef = tmp1*segmentParam[0].x*ly + tmp2*segmentParam[1].y*lx;
	float n2Coef = tmp2*segmentParam[0].y*ly + tmp1*segmentParam[1].x*lx;
	float n3mCoef = tmp2*segmentParam[0].y*lx;
	float m3nCoef = tmp2*segmentParam[1].y*ly;
	
	float tmpKd = vertKd*2.0;
	
	vec4 k1, k2, k3;
	k1 = vec4( mnCoef, m2Coef, n2Coef, 1.0);
	k2 = vec4( n3mCoef, m3nCoef, lx*(segmentParam[0].x+segmentParam[1].x+tmpKd), 1.0);
	k3 = vec4( tmp3*tmpKd*lx,  tmp3*tmpKd*ly,  ly*(segmentParam[0].x+segmentParam[1].x+tmpKd), 1.0);
	
	float tmp = 1.0f/4.0f;
	
	if ( end0>0.5 || end1>0.5 )
		tmp = tmp * 2.0f;
		
	gl_FragData[0] = tmp * k1;
	gl_FragData[1] = tmp * k2;
	gl_FragData[2] = tmp * k3;
}
