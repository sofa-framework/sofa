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

#ifndef __CONVERTTYPE_H_
#define __CONVERTTYPE_H_

#include <assert.h>
#include <vector>
#include <list>

namespace CGoGN {

namespace Utils {

/**
 * Create a ref to a type from a ref from another type
 * No copy only casting. No need to used T_IN template parameter
 * @param vec input ref
 * @ return a ref on same object with T_OUT type
 */
template <typename T_OUT, typename T_IN>
inline T_OUT& convertRef(T_IN& vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return *(reinterpret_cast<T_OUT*>(&vec));
}

/**
 * Create a const ref to a type from a const ref from another type
 * No copy only casting. No need to used T_IN template parameter
 * @param vec input ref
 * @ return a ref on same object with T_OUT type
 */
template <typename T_OUT, typename T_IN>
inline const T_OUT& convertRef(const T_IN& vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return *(reinterpret_cast<const T_OUT*>(&vec));
}

/**
 * Create a ptr of a type from a ptr of another type
 * Just a reinterpret cast in fact
 * @param vec input ptr
 * @return a ptr on same object with T_OUT type
 */
template <typename T_OUT, typename T_IN>
inline T_OUT* convertPtr(T_IN* vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return reinterpret_cast<T_OUT*>(vec);
}

/**
 * Create a const ptr of a type from a const ptr of another type
 * Just a reinterpret cast in fact
 * @param vec input ptr
 * @return a ptr on same object with T_OUT type
 */
template <typename T_OUT, typename T_IN>
inline const T_OUT* convertPtr(const T_IN* vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return reinterpret_cast<const T_OUT*>(vec);
}




template <typename T_OUT, typename T_IN>
inline std::vector<T_OUT>& convertVector(std::vector<T_IN>& vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return *(reinterpret_cast< std::vector<T_OUT>* >(&vec));
}


template <typename T_OUT, typename T_IN>
inline const std::vector<T_OUT>& convertVector(const std::vector<T_IN>& vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return *(reinterpret_cast< const std::vector<T_OUT>* >(&vec));
}





template <typename T_OUT, typename T_IN>
inline const std::list<T_OUT>& convertList(const std::list<T_IN>& vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return *(reinterpret_cast< const std::list<T_OUT>* >(&vec));
}


template <typename T_OUT, typename T_IN>
inline std::list<T_OUT>& convertList(std::list<T_IN>& vec)
{
	assert(sizeof(T_IN) == sizeof(T_OUT) || "incompatible size cast");
	return *(reinterpret_cast< std::list<T_OUT>* >(&vec));
}


} // namespace Utils

} // namespace CGoGN



#endif /* CONVERTTYPE_H_ */
