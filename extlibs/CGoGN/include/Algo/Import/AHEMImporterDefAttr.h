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

#include "Geometry/matrix.h"
#include "Algo/Import/AHEM.h"

namespace CGoGN
{

namespace Algo
{

namespace Import
{



/*
 *	Universal adaptor for standard attribute loaders that should work on every topological entity.
 */

template<typename MapType, typename AttrTypeLoader>
class UniversalLoader : public AttributeImporter<MapType>
{
public:
	UniversalLoader()								{}
	virtual ~UniversalLoader()						{}

	virtual bool Handleable(const AHEMAttributeDescriptor* ad) const;

	virtual void ImportAttribute(	MapType& map,
									const unsigned int* verticesId,
									const Dart* facesId,
									const AHEMHeader* hdr,
									const char* attrName,
									const AHEMAttributeDescriptor* ad,
									const void* buffer) const;

	void UnpackOnVertex(MapType& map, const unsigned int* verticesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const;
	void UnpackOnFace(MapType& map, const Dart* facesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const;
	void UnpackOnHE(MapType& map, const Dart* facesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const;
	void UnpackOnHEFC(MapType& map, const Dart* facesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const;
};


class Vec3FloatLoader;
class Mat44FloatLoader;


} // namespace Import

} // namespace Algo

} // namespace CGoGN
