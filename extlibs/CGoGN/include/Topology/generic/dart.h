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

#ifndef DART_H_
#define DART_H_

#include <iostream>
#include <string>

namespace CGoGN
{

const unsigned int INVALID_INDEX = 0xffffffff ;
const unsigned int EMBNULL = 0xffffffff;
const unsigned int MRNULL = 0xffffffff;

const unsigned int NB_THREAD = 16;

// DO NOT MODIFY (ORBIT_IN_PARENT function in Map classes)

const unsigned int NB_ORBITS	= 11;

const unsigned int DART			= 0;

const unsigned int VERTEX		= 1;
const unsigned int EDGE			= 2;
const unsigned int FACE			= 3;
const unsigned int VOLUME		= 4;
const unsigned int CC			= 5;

const unsigned int VERTEX1		= 6;
const unsigned int EDGE1		= 7;

const unsigned int VERTEX2		= 8;
const unsigned int EDGE2		= 9;
const unsigned int FACE2		= 10;

struct Dart
{
	unsigned int index;
	Dart(): index(0xffffffff) {}
	static Dart nil() { Dart d; d.index = 0xffffffff; return d; }
	static Dart create(unsigned int i) { Dart d; d.index = i; return d; }
	explicit Dart(unsigned int v): index(v) {}
	bool isNil() const { return index == 0xffffffff ; }
	/**
	 * affectation operator
	 * @param d the dart to store in this
	 */
	Dart operator=(Dart d) { index = d.index; return *this; }
	/**
	 * equality operator
	 * @param d the dart to compare with
	 */
	bool operator==(Dart d) const { return d.index == index; }
	/**
	 * different operator
	 * @param d the dart to compare with
	 */
	bool operator!=(Dart d) const { return d.index != index; }
	/**
	 * less operator, can be used for sorting
	 * @param d the dart to compare with
	 */
	bool operator<(Dart d) const { return index < d.index; }

	friend std::ostream& operator<<( std::ostream &out, const Dart& fa ) { return out << fa.index; }
	friend std::istream& operator>>( std::istream &in, Dart& fa ) { in >> fa.index; return in; }


	/**
	 * CGoGN name
	 */
	static std::string CGoGNnameOfType() { return "Dart"; }

	/**
	 * label is the index (cleaner that use d.index outside of maps
	 */
	unsigned int label() { return index; }
};

const Dart NIL = Dart::nil();

template<unsigned int ORBIT>
inline std::string orbitName()
{
    switch(ORBIT)
	{
	case DART:
		return "DART";
		break;
	case VERTEX:
		return "VERTEX";
		break;
	case EDGE:
		return "EDGE";
		break;
	case FACE:
		return "FACE";
		break;
	case VOLUME:
		return "VOLUME";
		break;
	case CC:
		return "CC";
		break;
	case VERTEX1:
		return "VERTEX1";
		break;
	case EDGE1:
		return "EDGE1";
		break;
	case VERTEX2:
		return "VERTEX2";
		break;
	case EDGE2:
		return "EDGE2";
		break;
	case FACE2:
		return "FACE2";
	default:
		break;

	}
	return "UNKNOWN";
}


}

#endif /* DART_H_ */
