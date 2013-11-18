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

#ifndef _MARKER_H_
#define _MARKER_H_

#include "Utils/mark.h"

namespace CGoGN
{

//! Class that manages the locking (lock and release) of markers
class MarkSet : public Mark
{
public:
	//! Constructor
	MarkSet() : Mark()
	{}

	//! Get a free marker and lock it
	Mark getNewMark()
	{
		Mark m;
		m.m_bits = 1;
		for (unsigned int i = 0; i < getNbMarks(); ++i)
		{
			if (!testMark(m))
			{
				setMark(m);
				return m;
			}
			m.m_bits <<= 1;
		}
		assert(!"No more markers");
		m.m_bits = 0;
		return m;
	}
	
	//! Release a locked marker
	void releaseMark(Mark m)
	{
		unsetMark(m);
	}
};

} //namespace CGoGN

#endif
