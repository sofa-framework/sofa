#version 120

uniform sampler2D texK[3];
uniform sampler2D texOcpy;

uniform vec2 winSize;
uniform float maxErr;
uniform float cellWidth;
uniform float cellHeight;

float getDeltaVol(float k[8], float m, float n)
{
	return k[1]*m*m + k[2]*n*n + k[3]*m*n + k[4]*pow(n,3)/m + k[5]*pow(m,3)/n + k[6]*n + k[7]*m;
}

void iterateM(inout float m, inout float n, float k[8], float expErr)
{
	float curErr = getDeltaVol(k, m, n);
	float deltaErr = expErr - curErr;
	float dvdm;

	for(int it=0; it<10; it++)
	{
		if (deltaErr>(-1e-4))
			break;

		dvdm =  2*k[1]*m + k[3]*n - k[4]*pow(n,3)/m/m + 3.0f*k[5]*m*m/n + k[7];
		m = m + deltaErr*dvdm;
		
		curErr = getDeltaVol(k, m, n);
		deltaErr = expErr - curErr;
	}
}

void iterateN(inout float m, inout float n, float k[8], float expErr)
{
	float curErr = getDeltaVol(k, m, n);
	float deltaErr = expErr - curErr;
	float dvdn;

	for(int it=0; it<10; it++)
	{
		if (deltaErr>(-1e-4))
			break;

		dvdn =  2*k[2]*n + k[3]*m - k[5]*pow(m,3)/n/n + 3.0f*k[4]*n*n/m + k[6];
		n = n + deltaErr*dvdn;

		curErr = getDeltaVol(k, m, n);
		deltaErr = expErr - curErr;
	}
}

bool isZero(float val)
{
	return abs(val)<1e-2;
}

void distributeRaysInCell(inout float m, inout float n, float k[8], float lx, float ly, float expErr)
{
	if ( isZero(lx) )
	{
		n = cellHeight;

		if( isZero(k[1]) &&  isZero(k[7]))
			m = cellWidth;
		else if( isZero(k[7]) )
			m = sqrt( expErr/k[1] );
		else if( isZero(k[1]) )
			m = expErr/k[7];
		else
		{
			float tmp1 = sqrt(k[7]*k[7]+4*expErr*k[1]);
			m = 0.5 * (tmp1 - k[7])/k[1];
		}

		if ( m > cellWidth )
			iterateM(m, n, k, expErr);
	}

	else if ( isZero(ly) )
	{
		m = cellWidth;

		if( isZero(k[2]) &&  isZero(k[6]))
			n = cellHeight;
		else if( isZero(k[6]) )
			n = sqrt( expErr/k[2] );
		else if( isZero(k[2]) )
			n = expErr/k[6];
		else
		{
			float tmp1 = sqrt(k[6]*k[6]+4*expErr*k[2]);
			n = 0.5 * (tmp1 - k[6])/k[2];
		}

		if ( n > cellHeight )
			iterateN(m, n, k, expErr);
	}
	else
	{
		if ( lx<=ly )
		{
			float tk = lx/ly;
			float a = k[1]*tk*tk + k[2] + k[3]*tk + k[4]/tk + k[5]*pow(tk,3);
			float b = k[6] + k[7]*tk;
			float c = -1 * expErr;

			if( isZero(a) &&  isZero(b))
			{
				n = cellHeight;
				m = cellWidth;
			}
			else
			{
				if( isZero(a) )
					n = -c / b;
				else if( isZero(b))
					n = sqrt( -c/a );
				else 
				{
					n = (-b + sqrt(b*b-4*a*c))/(2*a);
				}

				if( n>cellHeight )
				{
					m = tk*n;
					n = cellHeight;
					iterateM(m, n, k, expErr);
				}
				else
					m = tk*n;
			}

		}
		else
		{
			float tk = ly/lx;
			float a = k[1] + k[2]*tk*tk + k[3]*tk + k[4]*pow(tk,3) + k[5]/tk;
			float b = k[6]*tk + k[7];
			float c = -1 * expErr;

			if( isZero(a) &&  isZero(b))
			{
				n = cellHeight;
				m = cellWidth;
			}
			else
			{
				if( isZero(a) )
					m = -c / b;
				else if( isZero(b))
					m = sqrt( -c/a );
				else 
				{
					m = (-b + sqrt(b*b-4*a*c))/(2*a);
				}

				if( m>cellWidth )
				{
					n = tk*m;
					m = cellWidth;
					iterateN(m, n, k, expErr);
				}
				else
					n = tk*m;
			}
		}
	}
}

void main() 
{
	vec2 texCoord = vec2(gl_FragCoord.x/winSize.x, gl_FragCoord.y/winSize.y);
	vec4 edgeData[3];
	for(int i=0; i<3; i++)
		edgeData[i] = texture2D(texK[i], texCoord);

	vec4 ocpyData = texture2D(texOcpy, texCoord);

	vec2 rayNum = vec2(0, 0);
	if ( ocpyData.a > 0.1f )
		rayNum = vec2(1, 1);

	if( edgeData[0].a > 0.1f )
	{
		float lx = edgeData[1].z;
		float ly = edgeData[2].z;
		float k[8] = float[](0.0f, edgeData[0].y, edgeData[0].z, edgeData[0].x, 
			edgeData[1].x, edgeData[1].y, edgeData[2].x, edgeData[2].y);

		float m, n;
		distributeRaysInCell(m, n, k, lx, ly, maxErr);

		rayNum.x = max(1, cellWidth / m);
		rayNum.y = max(1, cellHeight / n);
	}

	gl_FragData[0] = vec4(rayNum, 0, 0);
}
