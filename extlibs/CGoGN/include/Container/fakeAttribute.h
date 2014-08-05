/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009, IGG Team, LSIIT, University of Strasbourg                *
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
* Web site: http://cgogn.unistra.fr/                                  *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef FAKEATTRIBUTE_H_
#define FAKEATTRIBUTE_H_

#include <fstream>

namespace CGoGN
{

/**
 * Ajoute tout ce qu'il faut a un type
 * qui n'a pas de nom
 * pour qu'il soit utilisable (ca compile!)
 */
template <typename T>
class NoTypeNameAttribute : public T
{
public:
	NoTypeNameAttribute() : T() {}
	NoTypeNameAttribute(int /*i*/) : T() {}

	NoTypeNameAttribute(const T& att) : T(att) {}
	NoTypeNameAttribute<T>& operator = (const T& fa) { return *this = NoTypeNameAttribute<T>(fa); }

	static std::string CGoGNnameOfType() { return ""; }
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const NoTypeNameAttribute<T>&)
{
	out << "no_output" ;
	return out ;
}

} // namespace CGoGN

#endif /* FAKEATTRIBUTE_H_ */
