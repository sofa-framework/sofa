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

namespace CGoGN
{

template <typename DART>
bool tXMap<DART>::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread)
{
	DartMarkerStore m(*this,thread);
	bool found = false;
	std::list<Dart> darts_list;
	darts_list.push_back(d);
	m.mark(d);

	typename std::list<Dart>::iterator prem = darts_list.begin();

	while (!found && prem != darts_list.end())
	{
		Dart d1 = *prem;

		// add phi21 and phi23 successor of they are not yet marked
		Dart d2 = phi1(d1); // turn in face
		Dart d3 = phi2(d1); // change volume
		Dart d4 = phi3(d1); // change volume

		if (!m.isMarked(d2))
		{
			darts_list.push_back(d2);
			m.mark(d2);
		}
		if (!m.isMarked(d3))
		{
			darts_list.push_back(d2);
			m.mark(d2);
		}
		if (!m.isMarked(d4))
		{
			darts_list.push_back(d4);
			m.mark(d4);
		}
		prem++;

		found =  f(d1);	// functor say finish
	}

	return found;
}

template <typename DART>
bool tXMap<DART>::foreach_dart_of_face(Dart d, FunctorType& f, unsigned int thread)
{
	if (foreach_dart_of_oriented_face(d,f,thread)) return true;

	Dart d3 = phi3(d);
	if (d3 != d) return foreach_dart_of_oriented_face(d3,f,thread);
	return false;
}

template <typename DART>
bool tXMap<DART>::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int thread)
{
	Dart dNext = d;
	do {
		if (tMap2<DART>::foreach_dart_of_edge(dNext,f,thread)) return true;
		dNext = alpha(2,dNext);
	} while (dNext != d);
	return false;
}

template <typename DART>
bool tXMap<DART>::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread)
{
	DartMarkerStore m(*this,thread) ;
	bool found = false ;

	std::list<Dart> darts_list ;
	darts_list.push_back(d) ;
	m.mark(d) ;

	typename std::list<Dart>::iterator prem = darts_list.begin() ;

	while (!found && prem != darts_list.end())
	{
		Dart d1 = *prem ;

		// add phi21 and phi23 successor if they are not yet marked
		Dart d3 = phi2(d1) ;
		Dart d2 = phi1(d3) ; // turn in volume
		Dart d4 = phi3(d3) ; // change volume

		if (!m.isMarked(d2))
		{
			darts_list.push_back(d2) ;
			m.mark(d2) ;
		}
		if ((d4!=d3) && !m.isMarked(d4))
		{
			darts_list.push_back(d4) ;
			m.mark(d4) ;
		}
		prem++ ;

		found = f(d1) ;
	}

	return found ;
}

} // namespace CGoGN
