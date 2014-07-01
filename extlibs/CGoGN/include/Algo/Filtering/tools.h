/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#include "Geometry/distances.h"

#include <time.h>
#include <stdlib.h>
#include <limits>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Filtering
{

template <typename PFP>
float computeHaussdorf(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& originalPosition, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::VEC3 VEC3 ;

	float dist_o = 0.0f ;
	float dist_f = 0.0f ;

	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
	{
		const VEC3& posO = originalPosition[d] ;
		const VEC3& posF = position2[d] ;

		float dv_o = std::numeric_limits<float>::max();
		float dv_f = std::numeric_limits<float>::max();

		// just test the faces around the vertex => warning not real haussdorff distance!
		Traversor2VF<typename PFP::MAP> tf(map, d) ;
		for(Dart it = tf.begin(); it != tf.end(); it = tf.next())
		{
			Dart e = map.phi1(it) ;
			const VEC3& Bo = originalPosition[e] ;
			const VEC3& Bf = position2[e] ;
			e = map.phi1(e) ;
			const VEC3& Co = originalPosition[e] ;
			const VEC3& Cf = position2[e] ;

			float d = Geom::squaredDistancePoint2Triangle(posO, posF, Bf, Cf) ;
			if(d < dv_o)
				dv_o = d ;
			d = Geom::squaredDistancePoint2Triangle(posF, posO, Bo, Co) ;
			if(d < dv_f)
				dv_f = d ;
		}

		if(dv_o > dist_o)
			dist_o = dv_o ;
		if(dv_f > dist_f)
			dist_f = dv_f ;
	}

	if (dist_f > dist_o)
		return sqrtf(dist_f) ;
	return sqrtf(dist_o) ;
}

template <typename PFP>
void computeNoise(typename PFP::MAP& map, long amount, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal)
{
	typedef typename PFP::VEC3 VEC3 ;

	// init the seed for random
	srand(time(NULL)) ;

	// apply noise on each vertex
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
	{
		const VEC3& pos = position[d] ;
		VEC3 norm = normal[d] ;

		float r1 = float(rand() % amount) / 100.0f ;
		float r2 = 0 ;
		if (amount >= 5)
			r2 = float(rand() % (amount/5)) / 100.0f ;

		long sign = rand() % 2 ;
		if (sign == 1) norm *= -1.0f ;
		float avEL = 0.0f ;
		VEC3 td(0) ;

		long nbE = 0 ;
		Traversor2VVaE<typename PFP::MAP> tav(map, d) ;
		for(Dart it = tav.begin(); it != tav.end(); it = tav.next())
		{
			const VEC3& p = position[it] ;
			VEC3 vec = p - pos ;
			float el = vec.norm() ;
			vec *= r2 ;
			td += vec ;
			avEL += el ;
			nbE++ ;
		}

		avEL /= float(nbE) ;
		norm *= avEL * r1 ;
		norm += td ;
		position2[d] = pos + norm ;
	}
}

//Uniform-distributed additive noise
//TODO do not touch to boundary vertices
template <typename PFP>
void computeUnfirmAdditiveNoise(typename PFP::MAP& map, float noiseIntensity, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::VEC3 VEC3 ;

	// Compute mesh center
	unsigned int count = 0;
	VEC3 centroid(0.0);

	TraversorV<typename PFP::MAP> tv(map) ;
	for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
	{
		centroid += position[dit];
		++count;
	}
	centroid /= count;

	double distanceCentroid = 0.0;

	// Compute the average distance from vertices to mesh center
	for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
	{
		VEC3 dist = position[dit];
		dist -= centroid;
		distanceCentroid += float(dist.norm());
	}
	distanceCentroid /= count;

	// add random uniform-distributed (between [-noiseLevel, +noiseLevel])
	srand((unsigned)time(NULL));
    float noisex, noisey, noisez;
    float noiseLevel = distanceCentroid * noiseIntensity;


    for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
	{
    	noisex = noiseLevel * (1.0*rand()/RAND_MAX-0.5)*2;
    	noisey = noiseLevel * (1.0*rand()/RAND_MAX-0.5)*2;
    	noisez = noiseLevel * (1.0*rand()/RAND_MAX-0.5)*2;

    	position2[dit] = position[dit] + VEC3(noisex,noisey,noisez);
	}
}


//Gaussian-distributed additive noise
//TODO do not touch to boundary vertices
template <typename PFP>
void computeGaussianAdditiveNoise(typename PFP::MAP& map, float noiseIntensity, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::VEC3 VEC3 ;

	// Compute mesh center
	unsigned int count = 0;
	VEC3 centroid(0.0);

	TraversorV<typename PFP::MAP> tv(map) ;
	for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
	{
		centroid += position[dit];
		++count;
	}
	centroid /= count;

	double distanceCentroid = 0.0;

	// Compute the average distance from vertices to mesh center
	for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
	{
		VEC3 dist = position[dit];
		dist -= centroid;
		distanceCentroid += float(dist.norm());
	}
	distanceCentroid /= count;

	// add random gaussian-distributed
	srand((unsigned)time(NULL));
    float noisex, noisey, noisez;
    float gaussNumbers[3];
    float noiseLevel = distanceCentroid * noiseIntensity;

    for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
	{

    	// pseudo-random Gaussian-distributed numbers generation from uniformly-distributed pseudo-random numbers
    	float x, y, r2;
    	for (int i=0; i<3; i++)
    	{
    		do
    		{
    			x = -1.0 + 2.0 * 1.0*rand()/RAND_MAX;
    			y = -1.0 + 2.0 * 1.0*rand()/RAND_MAX;
    			r2 = x * x + y * y;
    		} while ((r2>1.0)||(r2==0.0));
    		gaussNumbers[i] = y * sqrt(-2.0 * log(r2) / r2);
    	}

    	noisex = noiseLevel * gaussNumbers[0];
    	noisey = noiseLevel * gaussNumbers[1];
    	noisez = noiseLevel * gaussNumbers[2];

    	position2[dit] = position[dit] + VEC3(noisex,noisey,noisez);
	}

}

} //namespace Filtering

} //namespace Surface


namespace Volume
{

namespace Filtering
{

template <typename PFP>
void computeNoise(typename PFP::MAP& map, long amount, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::VEC3 VEC3 ;

	// add random gaussian-distributed
	srand((unsigned)time(NULL));
    float noisex, noisey, noisez;
    float gaussNumbers[3];
    float noiseLevel = 0.1f;

//    TraversorV<typename PFP::MAP> tv(map) ;
//    for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
//	{

//    	// pseudo-random Gaussian-distributed numbers generation from uniformly-distributed pseudo-random numbers
//    	float x, y, r2;
//    	for (int i=0; i<3; i++)
//    	{
//    		do
//    		{
//    			x = -1.0 + 2.0 * 1.0*rand()/RAND_MAX;
//    			y = -1.0 + 2.0 * 1.0*rand()/RAND_MAX;
//    			r2 = x * x + y * y;
//    		} while ((r2>1.0)||(r2==0.0));
//    		gaussNumbers[i] = y * sqrt(-2.0 * log(r2) / r2);
//    	}

//    	noisex = noiseLevel * gaussNumbers[0];
//    	noisey = noiseLevel * gaussNumbers[1];
//    	noisez = noiseLevel * gaussNumbers[2];

//    	position2[dit] = position[dit] + VEC3(noisex,noisey,noisez);
//	}


    // init the seed for random
    srand(time(NULL)) ;

    // apply noise on each vertex
    TraversorV<typename PFP::MAP> t(map) ;
    for(Dart d = t.begin(); d != t.end(); d = t.next())
    {
        const VEC3& pos = position[d] ;
        VEC3 norm = position[d] ;

        float r1 = float(rand() % amount) / 100.0f ;
        float r2 = 0 ;
        if (amount >= 5)
            r2 = float(rand() % (amount/5)) / 100.0f ;

        long sign = rand() % 2 ;
        if (sign == 1) norm *= -1.0f ;
        float avEL = 0.0f ;
        VEC3 td(0) ;

        long nbE = 0 ;
        Traversor3VVaE<typename PFP::MAP> tav(map, d) ;
        for(Dart it = tav.begin(); it != tav.end(); it = tav.next())
        {
            const VEC3& p = position[it] ;
            VEC3 vec = p - pos ;
            float el = vec.norm() ;
            vec *= r2 ;
            td += vec ;
            avEL += el ;
            nbE++ ;
        }

        avEL /= float(nbE) ;
        norm *= avEL * r1 ;
        norm += td ;
        position2[d] = pos + norm ;
    }
}


template <typename PFP>
void computeNoiseGaussian(typename PFP::MAP& map, long amount, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
    typedef typename PFP::VEC3 VEC3 ;

    // add random gaussian-distributed
    srand((unsigned)time(NULL));
    float noisex, noisey, noisez;
    float gaussNumbers[3];
    float noiseLevel = 0.08f;

    TraversorV<typename PFP::MAP> tv(map) ;
    for(Dart dit = tv.begin(); dit != tv.end(); dit = tv.next())
    {

        // pseudo-random Gaussian-distributed numbers generation from uniformly-distributed pseudo-random numbers
        float x, y, r2;
        for (int i=0; i<3; i++)
        {
            do
            {
                x = -1.0 + 2.0 * 1.0*rand()/RAND_MAX;
                y = -1.0 + 2.0 * 1.0*rand()/RAND_MAX;
                r2 = x * x + y * y;
            } while ((r2>1.0)||(r2==0.0));
            gaussNumbers[i] = y * sqrt(-2.0 * log(r2) / r2);
        }

        noisex = noiseLevel * gaussNumbers[0];
        noisey = noiseLevel * gaussNumbers[1];
        noisez = noiseLevel * gaussNumbers[2];

        position2[dit] = position[dit] + VEC3(noisex,noisey,noisez);
    }

}

} //namespace Filtering

} //namespace Volume

} //namespace Algo

} //namespace CGoGN
