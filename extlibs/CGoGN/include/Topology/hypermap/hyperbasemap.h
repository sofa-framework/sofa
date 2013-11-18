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

#ifndef __BASEHYPERMAP_H__
#define __BASEHYPERMAP_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <list>
#include <vector>

#include "Topology/generic/genericmap.h"

namespace CGoGN
{



template < typename DART >
class tBaseHyperMap : public tGenericMap<DART>
{
protected:
	static tBaseHyperMap<DART> m_global_map;

public:
	typedef typename tGenericMap<DART>::Dart Dart;

	tBaseHyperMap() {}
	
	tBaseHyperMap(const tBaseHyperMap<DART>& bm) : tGenericMap<DART>(bm) {}

	~tBaseHyperMap()
	{
		while (this->begin() != this->end())
		{
			deleteDart(this->begin(),false);
		}
	}

	Dart nil() { return m_global_map.end();}


	//! Test if the dart is free of topological links
	/*!	@param d an iterator to the current dart
	 */
	bool isFree(Dart d) 
	{
		for (unsigned i=1; i<=DART::nbRelations(); ++i)
			if (alpha(i,d) != d) 
				return false;

		return true;
	}
	
	void deleteDart(Dart d, bool warning=true)
	{
		if (warning && !isFree(d))
		{
			CGoGNout << "Warning: erasing a linked dart" << CGoGNendl;
		}
		tGenericMap<DART>::deleteDart(d);
	}

	Dart newDart() 
	{
		Dart d = tGenericMap<DART>::newDart();
		for (unsigned i=0; i< DART::nbStoredRelations(); ++i)
			d->m_topo[i] = nil();
		return d;
	}

	//! Get alpha relation
	Dart alpha( int i, Dart a) const 
	{
		assert(i < DART::nbRelations() || !"Out of bounds");
		return a->getPermutation(i);
	}

	//! Get inverse alpha relation
	Dart alpha_inv(int i, Dart a) const 
	{
		assert(i < DART::nbRelations() || !"Out of bounds");
		return a->getPermutationInverse(i);
	}


	//! Link dart a to dart d with a topological relation
	void alphaSew(unsigned i, Dart a, Dart d)
	{
		assert(i < DART::nbRelations() || !"Out of bounds");
		if (alpha(i,a)!=d) 
		{
			DART::setPermutation(i,a,d);
		}
		else 
		{
			CGoGNerr << "Warning: darts already linked with alpha" << i << CGoGNendl;
		}
	}

	//! Unlink dart a from its successor in topological relation
	void alphaUnsew( unsigned i, Dart a)
	{
		assert(i < DART::nbRelations() || !"Out of bounds");
		a->unsetPermutation(i,nil());
	}


	Dart alpha0( const Dart d)
	{
		return alpha(0,d);
	}

	Dart alpha_0(const Dart d)
	{
		return alpha_inv(0,d);
	}

	Dart alpha1( const Dart d)
	{
		return alpha( 1,d);
	}

	Dart alpha_1(const Dart d)
	{
		return alpha_inv(1,d);
	}

	Dart alpha2( const Dart d)
	{
		return alpha(2,d);
	}

	Dart alpha_2(const Dart d)
	{
		return alpha_inv(2,d);
	}


	void alpha0Sew(Dart d, Dart e)
	{
		alphaSew(0,d,e);
	}

	void alpha1Sew(Dart d, Dart e)
	{
		alphaSew(1,d,e);
	}

	void alpha2Sew(Dart d, Dart e)
	{
		alphaSew(2,d,e);
	}


	void alpha0Unsew(Dart d)
	{
		alphaUnsew(0,d);
	}

	void alpha1Unsew(Dart d)
	{
		alphaUnsew(1,d);
	}

	void alpha2Unsew(Dart d)
	{
		alphaUnsew(2,d);
	}

	bool foreach_dart_of_vertex(Dart d, FunctorType& f) { return false;}
	bool foreach_dart_of_edge(Dart d, FunctorType& f) { return false;}
	bool foreach_dart_of_face(Dart d, FunctorType& f) { return false;}
	bool foreach_dart_of_volume(Dart d, FunctorType& f) { return false;}
	bool foreach_dart_of_cc(Dart d, FunctorType& f) { return false;}


};

template <typename DART>   tBaseHyperMap<DART>  tBaseHyperMap<DART>::m_global_map = tBaseHyperMap<DART>();

} //namespace CGoGN

#endif
