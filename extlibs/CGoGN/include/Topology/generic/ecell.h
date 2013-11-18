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

#ifndef ECELL_H_
#define ECELL_H_


#include <vector>
#include <map>

#include "Topology/generic/genericmap.h"
#include "Topology/generic/marker.h"

namespace CGoGN
{
/**
* ECellDart class
* Allow operation on a line of attribute in algorithm
* \warning: use a map and a container as static variable
* Use setMap and setContainer when changing the map
*
*/
template <int DIM>
class ECellDart
{
	protected:

	/// map (static)
	static GenericMap* s_map;

	///container ptr acces (static, all ECellDart)
	static AttributeContainer* s_cont;

	///id of cell
	unsigned int m_id;

	public:
	/**
	 * constructor without container ( previous constructor must have been called once before)
	 * @param i index of cell
	 */
	ECellDart(unsigned int i);

	/**
	 * constructor without container ( previous constructor must have be call once before)
	 * @param i index of cell
	 */
	ECellDart();

	/**
	 * copy constructor
	 * @param ec the ECellDart copy
	 */
	ECellDart(const ECellDart<DIM>& ec);

	/**
	 * Destructor
	 * Remove attributes line if tempo
	 */
	~ECellDart();

	/**
	 * assign container
	 */
	static void setContainer(AttributeContainer& cont);

	/**
	 * assign map
	 */
	static void setMap(GenericMap& map);

	/**
	 * affect initial value to all attributes of cell
	 */
	void  zero();

	/**
	 *  affectation operator
	 */
	void operator =(const ECellDart<DIM>& ec);

	/**
	 *  auto-addition operator
	 */
	void operator +=(const ECellDart<DIM>& ec);

	/**
	 *  auto-substraction operator
	 */
	void operator -=(const ECellDart<DIM>& ec);

	/**
	 *  auto-multiplication operator
	 */
	void operator *=(double a);

	/**
	 *  auto-division operator
	 */
	void operator /=(double a);

	/**
	 * linear interpolation
	 */
	void lerp(const ECellDart<DIM>& ec1, const ECellDart<DIM>& ec2, double a);

	/**
	 * + operator return a new ECellDart<DIM> (temporary)
	 */
	ECellDart<DIM> operator +(const ECellDart<DIM>& ec);

	/**
	 * - operator return a new ECellDart<DIM> (temporary)
	 */
	ECellDart<DIM> operator -(const ECellDart<DIM>& ec);

	/**
	 * * operator return a new ECellDart<DIM> (temporary)
	 */
	ECellDart<DIM> operator *(double a);

	/**
	 * / operator return a new ECellDart<DIM> (temporary)
	 */
	ECellDart<DIM> operator /(double a);

	/**
	 * fake [] operator in fact call the constructor
	 * allow to consider ECell as a vector of itself !
	 */
	ECellDart<DIM> operator[](Dart d);

	/**
	 * fake at operator in fact call the constructor
	 * allow to consider ECell as a vector of itself !
	 */
	ECellDart<DIM> at(unsigned int i);

	//		friend std::ostream& operator<<(std::ostream& s, ECellDart<DIM> e) {
	//			s_cont << e.output();
	//			return s;
	//		}
};




/**
 * Some typedef
 */
typedef ECellDart<0> EVertex;
typedef ECellDart<1> EEdge;
typedef ECellDart<2> EFace;
typedef ECellDart<3> EVolume;





} //namespace CGoGN

#include "ecell.hpp"

#endif /* ECELL_H_ */
