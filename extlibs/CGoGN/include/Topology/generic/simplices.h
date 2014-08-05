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

#ifndef __SIMPLICES_H_
#define __SIMPLICES_H_

#include <iostream>
#include "cells.h"
#include "Algo/Topo/simplex.h"

namespace CGoGN
{


template <unsigned int ORBIT>
class Simplex
{
public:
	Dart dart;
	/// emoty construtor
	Simplex(): dart() {}
	/// construtor from Dart
	inline Simplex(Dart d): dart(d) {}

	inline Simplex(Cell<ORBIT> c): dart(c.dart) {}
	/// copy constructor
	inline Simplex(const Simplex<ORBIT>& c): dart(c.dart) {}
	/// Cell cast operator
	inline operator Cell<ORBIT>() {return Cell<ORBIT>(dart);}
	/// Dart cast operator
	inline operator Dart() {return dart;}
	/// check if this simplex is really a simplex
	template <typename MAP>
	bool check(const MAP& map) const { return Algo::Topo::isSimplex<MAP,ORBIT>(map,dart);}
};


typedef Simplex<FACE>   Triangle;
typedef Simplex<VOLUME> Tetra;


}

#endif /* __SIMPLICES_H_ */
