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

#ifndef __MC_TABLE__
#define __MC_TABLE__

#include "type.h"
namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{

/**
* Table for speed enhancement of Marching Cube
* indices correspond with notation of original
* article (GC V21 N4 1987)
* @warning The table.h file must be included only one time
* because initialization of statics is made in the same
* file. It can not be included in the marching-cube files
* because it is template
*/
class  accelMCTable
{
public:
/**
* Code of edge 6 bits one for each face.
* A code can only have 2 bits set to one
* bit 0 X=0, bit 1 X=1, bit 2 Y=0, bit 3 Y=1, bit 4 Z=0, bit 5 Z=1
*/

static const unsigned char m_EdgeCode[12];


/**
* This table store edges configuration function of
* the cube configurations (2^8=256)
*/
static const short m_EdgeTable[256];


/**
* This table store triangles configuration function of
* the cube configurations (2^8=256). 
* Maximum number of triangles is 5 : 15 indices -> 16 values with the final -1
*/
//static const char m_TriTable[256][16];
static const char m_TriTable[256][16];


/**
* This table store neighbourhood triangles configuration function of
* the cube configurations
*/
//static const char m_NeighTable[256][16];
static const char m_NeighTable[256][16];

};


} // end namespace
} // end namespace
} // end namespace
}


#endif



