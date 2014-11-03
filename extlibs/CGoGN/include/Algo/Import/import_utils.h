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

#ifndef _IMPORT_UTIL_H
#define _IMPORT_UTIL_H

#include <iostream>
#include <vector>

namespace Import
{

/**
* read the next property definition in PLY header
* @param fp file stream
* @param s1 first word following property (contain the type)
* @return true if a property has been readen. If not s1 contain the readen word
*/
bool readNextProperty(std::ifstream& fp, std::string& s1);

/**
* read PLY header
* @param fp file stream
* @param nbv number of vertices (output)
* @param nbf number of faces (output)
* @param binary true if ply is binary format (output)
* @param bigendian true if binary ply is big endian (old computer SGI SUN ...
* @param bulkVertex vector that contain number of byte to read for each unused vertex property
* @param bulkVertex vector that contain number of byte to read for each unused face property
*/
bool readHeaderPLY(std::ifstream& fp, long& nbv, long& nbf, long& nbs, bool& binary, bool& bigendian, std::vector<long>& bulkVertex,  std::vector<long>& bulkFace);

void big2littleEndian(unsigned char* ptr, long nb);

}

#endif

