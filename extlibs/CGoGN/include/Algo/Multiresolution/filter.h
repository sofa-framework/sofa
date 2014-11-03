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

#ifndef __MR_FILTERS__
#define __MR_FILTERS__

#include <cmath>

namespace CGoGN
{

namespace Algo
{

namespace MR
{

class Filter
{
public:
	Filter() {}
	virtual ~Filter() {}
	virtual void operator() () = 0 ;
} ;

template <typename PFP>
unsigned int vertexLevel(typename PFP::MAP& map, Vertex v)
{
    assert(map.getDartLevel(v) <= map.getCurrentLevel() || !"vertexLevel : called with a dart inserted after current level") ;

	unsigned int level = map.getMaxLevel();

	map.foreach_dart_of_orbit(v, [&] (Dart d)
	{
		unsigned int ldit = map.getDartLevel(d) ;
		if(ldit < level)
			level = ldit;
	});

//	Dart dit = d;
//	do
//	{
//		unsigned int ldit = map.getDartLevel(dit) ;
//		if(ldit < level)
//			level = ldit;
//
//		dit = map.phi2(map.phi_1(dit));
//	}
//	while(dit != d);

	return level;
}


template <typename PFP, typename T>
void filterLowPass(typename PFP::MAP& map, VertexAttribute<T, typename PFP::MAP>& attIn, unsigned int cutoffLevel)
{
	unsigned int cur = map.getCurrentLevel();
	unsigned int max = map.getMaxLevel();

	map.setCurrentLevel(max);

	TraversorV<typename PFP::MAP> tv(map);
	for (Dart d = tv.begin(); d != tv.end(); d = tv.next())
	{
		if(vertexLevel<PFP>(map,d) > cutoffLevel)
			attIn[d] = T(0.0);
	}

	map.setCurrentLevel(cur);
}

template <typename PFP, typename T>
void filterHighPass(typename PFP::MAP& map, VertexAttribute<T, typename PFP::MAP>& attIn, unsigned int cutoffLevel)
{
	unsigned int cur = map.getCurrentLevel();
	unsigned int max = map.getMaxLevel();

	map.setCurrentLevel(max);

	TraversorV<typename PFP::MAP> tv(map);
	for (Dart d = tv.begin(); d != tv.end(); d = tv.next())
	{
		if(vertexLevel<PFP>(map,d) < cutoffLevel)
			attIn[d] = T(0.0);
	}

	map.setCurrentLevel(cur);
}

template <typename PFP, typename T>
void filterBandPass(typename PFP::MAP& map, VertexAttribute<T, typename PFP::MAP>& attIn, unsigned int cutoffLevelLow, unsigned int cutoffLevelHigh)
{
	unsigned int cur = map.getCurrentLevel();
	unsigned int max = map.getMaxLevel();

	map.setCurrentLevel(max);

	TraversorV<typename PFP::MAP> tv(map);
	for (Dart d = tv.begin(); d != tv.end(); d = tv.next())
	{
		unsigned int vLevel = vertexLevel<PFP>(map,d);
		if(cutoffLevelLow > vLevel && vLevel < cutoffLevelHigh)
			attIn[d] = T(0.0);
	}

	map.setCurrentLevel(cur);
}

template <typename PFP>
typename PFP::VEC3 doTwist(typename PFP::VEC3 pos, float t )
{
	typedef typename PFP::VEC3 VEC3;

	float st = std::sin(t);
	float ct = std::cos(t);
	VEC3 new_pos;

	new_pos[0] = pos[0]*ct - pos[2]*st;
	new_pos[2] = pos[0]*st + pos[2]*ct;

	new_pos[1] = pos[1];

	return new_pos;
}

template <typename PFP>
void frequencyDeformation(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& attIn, unsigned int cutoffLevel)
{
	float time = 1.0;
	//float angle_deg_max = 0.4;
	//float height = 0.4;

	float A = 20.0;
	float frequency = 1.0;
	float phase = time * 2.0;

	TraversorV<typename PFP::MAP> tv(map);
	for (Dart d = tv.begin(); d != tv.end(); d = tv.next())
	{
		typename PFP::VEC3 p = attIn[d];

		float dist = std::sqrt(p[0]*p[0] + p[2]*p[2]);

		p[1] += A * std::sin(frequency * dist + phase);

//		float angle_deg = angle_deg_max * std::sin(time);
//		float angle_rad = angle_deg * 3.14159 / 180.0;

//		float ang = (height*0.5 + attIn[d][1])/height * angle_rad;

//		attIn[d] = doTwist<PFP>(attIn[d], ang);
	}
}

} // namespace MR

} // namespace Algo

} // namespace CGoGN

#endif
