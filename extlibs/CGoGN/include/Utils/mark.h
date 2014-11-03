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

#ifndef _MARK_H_
#define _MARK_H_

#include <cassert>
#include <string>
#include <iostream>
#include <sstream>

namespace CGoGN
{

//! A Mark object is a set of boolean flags.
/*! The values are stored as bits in a integral value.
 *  Each single bit means: 1 = YES / 0 = NO.
 *  Operation on bits are performed through classical AND / OR / XOR
 *  operators on the integral value. 
 */
class Mark
{
	friend class MarkSet;

	typedef unsigned int value_type;
	value_type m_bits;

public:
	//! Return the number of single marks (or bits) in a Mark
	static unsigned getNbMarks()
	{
		return 8 * sizeof(Mark);
	}

	//! Constructor
	Mark() : m_bits(0) {}

	Mark(unsigned int a) : m_bits(a) {}

	//! Copy constructor
	Mark(const Mark& m) : m_bits(m.m_bits) {}

	static std::string CGoGNnameOfType() { return "Mark"; }

	//! Logical OR on two marks
	Mark operator+(const Mark m) const
	{
		Mark n(*this);
		n.setMark(m);
		return n;
	}

	inline void clear()
	{
		m_bits = 0;
	}

	//! Test if all marks are set to NO (0)
	inline bool isClear()
	{
		return (m_bits == 0);
	}

	inline void invert()
	{
		m_bits = ~m_bits;
	}

	//! Set given marks to YES
	/*! @param m the marks to set
	 */
	inline void setMark(const Mark m)
	{
		m_bits |= m.m_bits;
	}

	//! Unset given marks or set them to NO
	/*! @param m the marks to unset
	 */
	inline void unsetMark(const Mark m)
	{
		m_bits &= ~m.m_bits;
	}

	//! Test if the given marks are set to YES
	/*! @param m the marks to test
	 */
	inline bool testMark(const Mark m) const
	{
		return (m_bits & m.m_bits) != 0;
	}

	//! Return the state of the mark as value_type
	inline value_type getMarkVal() const
	{
        return m_bits;
	}

	//! Save the binary state of mark in a string
	std::string getMarkAsBinaryString() const
	{
		std::ostringstream oss;
		// transforme la chaine en binaire (plus lisible)
		value_type v = m_bits;
		for (unsigned int i = 0; i < 8*sizeof(value_type); ++i)
		{
			oss << (v%2);
			v = v >> 1;
		}
		return oss.str();
	}

	//! Save the state of mark in a string
	std::string getMarkAsString() const
	{
		std::ostringstream oss;
		oss << m_bits;
		return oss.str();
	}

	//! Set marks from a saved state
	void setMarkVal(const value_type bits)
	{
        m_bits = bits;
	}

	//! Set marks from a saved state
	void setMarkVal(const std::string& state)
	{
		std::istringstream iss(state);
		iss >> m_bits;
	}

	//! Stream output operator
	friend std::ostream& operator<<(std::ostream& s, const Mark m)
	{
		s << m.m_bits;
		return s;
	}

	//! Stream output operator
	friend std::istream& operator>>(std::istream& s, Mark& m)
	{
		s >> m.m_bits;
		return s;
	}

	// math operator (fake, juste here to enable compilation)
	void operator +=(const Mark& /*m*/) {}
	void operator -=(const Mark& /*m*/) {}
	void operator *=(double /*a*/) {}
	void operator /=(double /*a*/) {}
};

} // namespace CGoGN

#endif
