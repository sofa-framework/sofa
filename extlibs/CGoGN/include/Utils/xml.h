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

#ifndef __CGoGN_XML__
#define __CGoGN_XML__

#include <string>
#include "Utils/cgognStream.h"
#include "tinyxml2.h"


namespace CGoGN
{


inline bool XMLisError(tinyxml2::XMLError err, const std::string& msg)
{
	if (err != tinyxml2::XML_NO_ERROR)
	{
		CGoGNerr << msg << CGoGNendl;
		return true;
	}
	return false;
}

inline std::string XMLAttribute(tinyxml2::XMLElement* node, const char* attName)
{
	const char *ptr = node->Attribute(attName);
	if (ptr == NULL)
	{
		CGoGNerr << "Warning attribute "<< attName << " not found"<< CGoGNendl;
		return "";
	}
	return std::string(ptr);
}

}
#endif
