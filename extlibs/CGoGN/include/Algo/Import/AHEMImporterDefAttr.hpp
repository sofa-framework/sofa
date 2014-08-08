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

template<typename MapType, typename AttrTypeLoader>
bool UniversalLoader<MapType, AttrTypeLoader>::Handleable(const AHEMAttributeDescriptor* ad) const
{
	return AttrTypeLoader::Handleable(ad);
}

template<typename MapType, typename AttrTypeLoader>
void UniversalLoader<MapType, AttrTypeLoader>::ImportAttribute(	MapType& map,
																const unsigned int* verticesId,
																const Dart* facesId,
																const AHEMHeader* hdr,
																const char* attrName,
																const AHEMAttributeDescriptor* ad,
																const void* buffer) const
{
	switch(ad->owner)
	{
	case AHEMATTROWNER_VERTEX:
		UnpackOnVertex(map, verticesId, hdr, attrName, buffer);
		break;

	case AHEMATTROWNER_FACE:
		UnpackOnFace(map, facesId, hdr, attrName, buffer);
		break;

	case AHEMATTROWNER_HALFEDGE:
		UnpackOnHE(map, facesId, hdr, attrName, buffer);
		break;

	case AHEMATTROWNER_HE_FACECORNER:
		UnpackOnHEFC(map, facesId, hdr, attrName, buffer);
		break;

	default:
		break;
	}
}

template<typename MapType, typename AttrTypeLoader>
void UniversalLoader<MapType, AttrTypeLoader>::UnpackOnVertex(MapType& map, const unsigned int* verticesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const
{
	VertexAttribute<typename AttrTypeLoader::ATTR_TYPE> attr = map.template getAttribute<typename AttrTypeLoader::ATTR_TYPE, VERTEX>(attrName);

	if (!attr.isValid())
		attr = map.template addAttribute<typename AttrTypeLoader::ATTR_TYPE, VERTEX>(attrName);

	char* p = (char*)buffer;

	for(unsigned int i = 0 ; i < hdr->meshHdr.vxCount ; i++)
	{
		AttrTypeLoader::Extract(&attr[verticesId[i]], p);
		p += AttrTypeLoader::TYPE_SIZE_IN_BUFFER;
	}
}

template<typename MapType, typename AttrTypeLoader>
void UniversalLoader<MapType, AttrTypeLoader>:: UnpackOnFace(MapType& map, const Dart* facesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const
{
	FaceAttribute<typename AttrTypeLoader::ATTR_TYPE> attr = map.template getAttribute<typename AttrTypeLoader::ATTR_TYPE>, FACE(attrName);

	if (!attr.isValid())
		attr = map.template addAttribute<typename AttrTypeLoader::ATTR_TYPE>(FACE, attrName);

	char* p = (char*)buffer;

	for(unsigned int i = 0 ; i < hdr->meshHdr.faceCount ; i++)
	{
		AttrTypeLoader::Extract(&attr[facesId[i]], p);
		p += AttrTypeLoader::TYPE_SIZE_IN_BUFFER;
	}
}

template<typename MapType, typename AttrTypeLoader>
void UniversalLoader<MapType, AttrTypeLoader>:: UnpackOnHE(MapType& map, const Dart* facesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const
{
	DartAttribute<typename AttrTypeLoader::ATTR_TYPE> attr = map.template getAttribute<typename AttrTypeLoader::ATTR_TYPE, DART>(attrName);

	if (!attr.isValid())
		attr = map.template addAttribute<typename AttrTypeLoader::ATTR_TYPE, DART>(attrName);

	char* p = (char*)buffer;

	for(unsigned int i = 0 ; i < hdr->meshHdr.faceCount ; i++)
	{
		Dart start = map.phi_1(facesId[i]);
		Dart d = start;

		do
		{
			AttrTypeLoader::Extract(&attr[d], p);
			p += AttrTypeLoader::TYPE_SIZE_IN_BUFFER;
			d = map.phi1(d);
		}
		while(d != start);
	}
}

template<typename MapType, typename AttrTypeLoader>
void UniversalLoader<MapType, AttrTypeLoader>:: UnpackOnHEFC(MapType& map, const Dart* facesId, const AHEMHeader* hdr, const char* attrName, const void* buffer) const
{
	DartAttribute<typename AttrTypeLoader::ATTR_TYPE> attr = map.template getAttribute<typename AttrTypeLoader::ATTR_TYPE, DART>(attrName);

	if (!attr.isValid())
		attr = map.template addAttribute<typename AttrTypeLoader::ATTR_TYPE, DART>(attrName);

	char* p = (char*)buffer;

	for(unsigned int i = 0 ; i < hdr->meshHdr.faceCount ; i++)
	{
		Dart d = facesId[i];

		do
		{
			AttrTypeLoader::Extract(&attr[d], p);
			p += AttrTypeLoader::TYPE_SIZE_IN_BUFFER;
			d = map.phi1(d);
		}
		while(d != facesId[i]);
	}
}

/*
 *	Final-glue code for universal parsing of 
 *	[float, float, float] -> Geom::Vector<3, float>
 *	attribute.
 *
 *	Works with UniversalLoader
 */
class Vec3FloatLoader
{
public:
	static const unsigned int TYPE_SIZE_IN_BUFFER = 12;
	typedef Geom::Vector<3, float> ATTR_TYPE;

	static inline bool Handleable(const AHEMAttributeDescriptor* ad)
	{
		return IsEqualGUID(ad->dataType, AHEMDATATYPE_FLOAT32) && ad->dimension == 3;
	}

	static inline void Extract(void* val, void* buffer)
	{
		float* v = (float*)buffer;
		*(ATTR_TYPE*)val = ATTR_TYPE(v[0], v[1], v[2]);
	}	
};


/*
 *	Final-glue code for universal parsing of 
 *	[float]^16 -> Geom::Matrix<4, 4, float> (column-major order / OpenGL-style)
 *	attribute
 *
 *	Works with UniversalLoader
 */
class Mat44FloatLoader
{
public:
	static const unsigned int TYPE_SIZE_IN_BUFFER = 64;
	typedef Geom::Matrix<4, 4, float> ATTR_TYPE;

	static inline bool Handleable(const AHEMAttributeDescriptor* ad)
	{
		return IsEqualGUID(ad->dataType, AHEMDATATYPE_FLOAT32) && ad->dimension == 16;
	}

	static inline void Extract(void* val, void* buffer)
	{
		float* v = (float*)buffer;

		Geom::Matrix<4, 4, float> m;

		m(0, 0) = v[0];
		m(1, 0) = v[1];
		m(2, 0) = v[2];
		m(3, 0) = v[3];

		m(0, 1) = v[4];
		m(1, 1) = v[5];
		m(2, 1) = v[6];
		m(3, 1) = v[7];

		m(0, 2) = v[8];
		m(1, 2) = v[9];
		m(2, 2) = v[10];
		m(3, 2) = v[11];

		m(0, 3) = v[12];
		m(1, 3) = v[13];
		m(2, 3) = v[14];
		m(3, 3) = v[15];

		*(ATTR_TYPE*)val = m;
	}	
};

} // namespace Import

} // namespace Algo

} // namespace CGoGN
