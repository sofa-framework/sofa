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

#ifndef __AHEMIMPORTER_H__
#define __AHEMIMPORTER_H__

#include <fstream>
#include "Algo/Import/AHEM.h"



namespace CGoGN
{

namespace Algo
{

namespace Import 
{


/*
 *	Actual loaders for attributes
 */
	
template<typename MapType>
class AttributeImporter
{
public:
#ifdef _WIN32
	virtual ~AttributeImporter() = 0			{}
#else
	virtual ~AttributeImporter()				{}
#endif

	virtual bool Handleable(const AHEMAttributeDescriptor* ad) const = 0;

	virtual void ImportAttribute(	MapType& map,
									const unsigned int* verticesId,
									const Dart* facesId,
									const AHEMHeader* hdr,
									const char* attrName,
									const AHEMAttributeDescriptor* ad,
									const void* buffer) const = 0;
};



static const unsigned int ATTRIBUTE_NOTFOUND = (unsigned int)-1;

/*
 *	Importer
 */

template<typename PFP>
class AHEMImporter
{
public:
	AHEMImporter(bool useDefaultImporters = true);
	~AHEMImporter();

	bool Open(typename PFP::MAP* m, const char* filename);
	void Close();
	
	void LoadMesh();
	bool LoadAttribute(unsigned int attrIx, const char* attrName, const AttributeImporter<typename PFP::MAP>* imp);
	bool LoadAttribute(const GUID& semantic, const char* attrName = NULL);
	void LoadAllAttributes(bool* status = NULL);

	// Low-level access to attributes

	inline unsigned int GetAttributesNum();
	inline void GetAttribute(AHEMAttributeDescriptor** ad, char** attrName, unsigned int ix);
	inline unsigned int FindAttribute(const GUID& semantic);

	// Attribute importers and helpers

	std::vector<AttributeImporter<typename PFP::MAP>*> loadersRegistry;

	inline std::vector<AttributeImporter<typename PFP::MAP>*> FindImporters(const GUID& attrSemanticId);
	inline std::vector<AttributeImporter<typename PFP::MAP>*> FindImporters(const AHEMAttributeDescriptor* ad);

protected:

	void LoadTopology();
	void LoadPosition(AHEMAttributeDescriptor* posDescr);


	typename PFP::MAP* map;
	std::ifstream f;

	AHEMHeader hdr;

	AHEMAttributeDescriptor* attrDesc;
	char** attrNames;

	char* buffer;
	unsigned int bufferSize;

	unsigned int* verticesId;
	Dart* facesId;
};




} // namespace Import

} // namespace Algo

} // namespace CGoGN

#include "Algo/Import/AHEMImporter.hpp"

#endif
