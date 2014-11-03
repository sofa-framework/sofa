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

#ifndef PARTBASE_H
#define PARTBASE_H

//#include "Geometry/vector_gen.h"

namespace CGoGN
{

namespace Algo
{

namespace MovingObjects
{

/* A particle base defines a position with a displacement function */

template <typename PFP>
class ParticleBase
{
protected:
	typename PFP::VEC3 m_position ;
	unsigned int m_state ;

public:
	ParticleBase(const typename PFP::VEC3& position) :
		m_position(position), m_state(FACE)
	{
	}

	virtual ~ParticleBase()
	{

	}

	void setState(unsigned int state)
	{
		m_state = state ;
	}

	unsigned int getState()
	{
		return m_state ;
	}

	bool move(const typename PFP::VEC3& position)
	{
		m_position = position ;
		return true ;
	}

	const typename PFP::VEC3& getPosition() const
	{
		return m_position ;
	}
} ;

} // namespace MovingObjects

} // namespace Algo

} // namespace CGoGN

#endif
