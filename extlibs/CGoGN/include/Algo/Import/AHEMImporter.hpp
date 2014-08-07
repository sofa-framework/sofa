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

#include "Topology/generic/attributeHandler.h"

#include "Geometry/matrix.h"

#include "Algo/Import/AHEMImporterDefAttr.h"


namespace CGoGN
{

namespace Algo
{

namespace Import
{


class HESorter
{
public:
	unsigned int vxIdTo;
	unsigned int vxIdFrom;

	Dart d;

	inline bool operator<(const HESorter& h) const
	{
		return (vxIdTo < h.vxIdTo) || (vxIdTo == h.vxIdTo && vxIdFrom < h.vxIdFrom);
	}
};



#ifdef _WIN32
int __cdecl OppositeSearch(const void* key, const void* datum)
#else
int OppositeSearch(const void* key, const void* datum)
#endif
{
	HESorter* k = (HESorter*)key;
	HESorter* d = (HESorter*)datum;

	int z = (int)k->vxIdFrom - (int)d->vxIdTo;
	return z != 0 ? z : ((int)k->vxIdTo - (int)d->vxIdFrom);
}



template<typename PFP>
AHEMImporter<PFP>::AHEMImporter(bool useDefaultImporters)
{
	map = NULL;

	attrDesc = NULL;
	attrNames = NULL;

	verticesId = NULL;
	facesId = NULL;


	// Default attribute loaders

	if(useDefaultImporters)
	{
		loadersRegistry.push_back(new UniversalLoader<typename PFP::MAP, Vec3FloatLoader>);
		loadersRegistry.push_back(new UniversalLoader<typename PFP::MAP, Mat44FloatLoader>);
	}
}


template<typename PFP>
AHEMImporter<PFP>::~AHEMImporter()
{
	if(f.is_open())
		Close();


	// Release all registered attribute loaders

	for(unsigned int i = 0 ; i < loadersRegistry.size() ; i++)
		delete loadersRegistry[i];
}
	
	
	
template<typename PFP>
bool AHEMImporter<PFP>::Open(typename PFP::MAP* m, const char* filename)
{
	// Halt if file is a file is already open

	if(f.is_open() || map)
		return false;


	// Open file

	f.open(filename, std::ios::binary);

	if (!f.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}


	// Keep track of current map

	map = m;


	// Read header

	f.read((char*)&hdr, sizeof(AHEMHeader));

	if(hdr.magic != AHEM_MAGIC)
		CGoGNerr << "Warning: " << filename << " invalid magic" << CGoGNendl;



	// Read attributes

	attrDesc = new AHEMAttributeDescriptor[hdr.attributesChunkNumber];
	attrNames = new char*[hdr.attributesChunkNumber];

	for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
	{
		f.read((char*)(attrDesc + i), sizeof(AHEMAttributeDescriptor));
		
		attrNames[i] = new char[attrDesc[i].nameSize + 1];
		f.read(attrNames[i], attrDesc[i].nameSize);
		attrNames[i][attrDesc[i].nameSize] = '\0';
	}
	

	// Compute buffer size for largest chunk and allocate

	bufferSize = hdr.meshHdr.meshChunkSize;

	for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
		if(attrDesc[i].attributeChunkSize > bufferSize)
			bufferSize = attrDesc[i].attributeChunkSize;


	buffer = new char[bufferSize];

	return true;
}


template<typename PFP>
void AHEMImporter<PFP>::Close()
{
	map = NULL;

	f.close();

	for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
		delete[] attrNames[i];

	delete[] attrNames;
	attrNames = NULL;

	delete[] attrDesc;
	attrDesc = NULL;

	delete[] buffer;
	buffer = NULL;

	delete[] verticesId;
	verticesId = NULL;

	delete[] facesId;
	facesId = NULL;
}


template<typename PFP>
void AHEMImporter<PFP>::LoadMesh()
{
	// Load mesh topological structure

	LoadTopology();


	// Load vertices position

	for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
		if(IsEqualGUID(attrDesc[i].semantic, AHEMATTRIBUTE_POSITION))
		{
			LoadPosition(attrDesc + i);
			break;
		}
}

template<typename PFP>
void AHEMImporter<PFP>::LoadTopology()
{
    // Allocate vertices

	AttributeContainer& vxContainer = map->template getAttributeContainer<VERTEX>();

	verticesId = new unsigned int[hdr.meshHdr.vxCount];

	for(unsigned int i = 0 ; i < hdr.meshHdr.vxCount ; i++)
		verticesId[i] = vxContainer.insertLine();

	// Ensure vertices are created by querying the position attribute

	VertexAttribute<typename PFP::VEC3> position =  map->template getAttribute<typename PFP::VEC3, VERTEX>("position") ;

	if (!position.isValid())
		position = map->template addAttribute<typename PFP::VEC3, VERTEX>("position") ;

	// Read faces stream and create faces [only intra-face links]
	
	HESorter* addedHE = new HESorter[hdr.meshHdr.heCount];

	facesId = new Dart[hdr.meshHdr.faceCount];

	f.read(buffer, hdr.meshHdr.meshChunkSize);
	char* batch = buffer;

	unsigned int fId = 0;
	unsigned int heId = 0;

	while(fId < hdr.meshHdr.faceCount)
	{
		AHEMFaceBatchDescriptor* fbd = (AHEMFaceBatchDescriptor*)batch;
		stUInt32* ix = (stUInt32*)(batch + sizeof(AHEMFaceBatchDescriptor));

		for(unsigned int i = 0 ; i < fbd->batchLength ; i++)
		{
			Dart d = map->newFace(fbd->batchFaceSize);							// create face
			facesId[fId++] = d;

			unsigned int firstVx = verticesId[*ix++];							// setup face's vertices up to last HE
			unsigned int prevVx = firstVx;

			for(unsigned int k = 0 ; k < fbd->batchFaceSize - 1 ; k++)
			{
				addedHE[heId].d = d;
				addedHE[heId].vxIdFrom = prevVx;
				addedHE[heId].vxIdTo = verticesId[*ix];

				map->setDartEmbedding<VERTEX>(d, prevVx);
				d = map->phi1(d);

				prevVx = *ix++;
				heId++;
			}

			// last HE

			addedHE[heId].d = d;
			addedHE[heId].vxIdFrom = prevVx;
			addedHE[heId].vxIdTo = firstVx;

			map->setDartEmbedding<VERTEX>(d, prevVx);
			heId++;
		}

		batch = (char*)ix;
	}

	// Sort the HE for fast retrieval

	std::sort(addedHE, addedHE + hdr.meshHdr.heCount);

	// Sew faces [inter-face links]

	for(unsigned int i = 0 ; i < hdr.meshHdr.heCount ; i++)
	{
		HESorter* opposite = (HESorter*)bsearch(addedHE + i, addedHE, hdr.meshHdr.heCount, sizeof(HESorter), OppositeSearch);

		if(opposite && opposite->d == map->phi2(opposite->d))
			map->sewFaces(addedHE[i].d, opposite->d);
	}
}

template <typename PFP>
void AHEMImporter<PFP>::LoadPosition(AHEMAttributeDescriptor* posDescr)
{
	VertexAttribute<typename PFP::VEC3> position =  map->template getAttribute<typename PFP::VEC3, VERTEX>("position") ;

	if (!position.isValid())
		position = map->template addAttribute<typename PFP::VEC3, VERTEX>("position") ;

	f.seekg(posDescr->fileStartOffset, std::ios_base::beg);
	f.read(buffer, posDescr->attributeChunkSize);

	float* q = (float*)buffer;

	for(unsigned int i = 0 ; i < hdr.meshHdr.vxCount ; i++)
	{
		position[verticesId[i]] = typename PFP::VEC3(q[0], q[1], q[2]);
		q += 3;
	}
}

template<typename PFP>
bool AHEMImporter<PFP>::LoadAttribute(unsigned int attrIx, const char* attrName, const AttributeImporter<typename PFP::MAP>* imp)
{
	if(attrIx >= hdr.attributesChunkNumber)
		return false;

	AHEMAttributeDescriptor* ad = attrDesc + attrIx;

	// Fill buffer from file data

	f.seekg(ad->fileStartOffset, std::ios_base::beg);
	f.read(buffer, ad->attributeChunkSize);

	// Import attribute

	imp->ImportAttribute(*map, verticesId, facesId, &hdr, attrName, ad, buffer);

	return true;
}

template<typename PFP>
bool AHEMImporter<PFP>::LoadAttribute(const GUID& semantic, const char* attrName)
{
	// Locate the descriptor using semantic identifier

	unsigned int ix = FindAttribute(semantic);

	if(ix == ATTRIBUTE_NOTFOUND)
		return false;

	AHEMAttributeDescriptor* ad = attrDesc + ix;
	const char* name = attrName ? attrName : attrNames[ix];

	// Find suitable importer

	std::vector<AttributeImporter<typename PFP::MAP>*> impList = FindImporters(ad);

	if(impList.empty())
		return false;

	AttributeImporter<typename PFP::MAP>* imp = impList[0];

	// Fill buffer from file data

	f.seekg(ad->fileStartOffset, std::ios_base::beg);
	f.read(buffer, ad->attributeChunkSize);

	// Import attribute

	imp->ImportAttribute(*map, verticesId, facesId, &hdr, name, ad, buffer);

	return true;
}

template<typename PFP>
void AHEMImporter<PFP>::LoadAllAttributes(bool* status)
{
	for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
	{
		std::vector<AttributeImporter<typename PFP::MAP>*> impList = FindImporters(attrDesc + i);

		if(!impList.empty())
			LoadAttribute(i, attrNames[i], impList[0]);

		if(status)
			status[i] = !impList.empty();
	}
}

template<typename PFP>
unsigned int AHEMImporter<PFP>::GetAttributesNum()
{
	return hdr.attributesChunkNumber;
}

template<typename PFP>
void AHEMImporter<PFP>::GetAttribute(AHEMAttributeDescriptor** ad, char** attrName, unsigned int ix)
{
	if(ix < hdr.attributesChunkNumber)
	{
		*ad = attrDesc + ix;
		*attrName = attrNames[ix];
	}
	else
	{
		*ad = NULL;
		*attrName = NULL;
	}
}

template<typename PFP>
unsigned int AHEMImporter<PFP>::FindAttribute(const GUID& semantic)
{
	for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
		if(IsEqualGUID(attrDesc[i].semantic, semantic))
			return i;

	return ATTRIBUTE_NOTFOUND;
}

template<typename PFP>
std::vector<AttributeImporter<typename PFP::MAP>*> AHEMImporter<PFP>::FindImporters(const GUID& attrSemanticId)
{
	std::vector<AttributeImporter<typename PFP::MAP>*> ret;

	unsigned int ix = FindAttribute(attrSemanticId);

	if(ix == ATTRIBUTE_NOTFOUND)
		return ret;

	return FindImporters(attrDesc + ix);
}

template<typename PFP>
std::vector<AttributeImporter<typename PFP::MAP>*> AHEMImporter<PFP>::FindImporters(const AHEMAttributeDescriptor* ad)
{
	std::vector<AttributeImporter<typename PFP::MAP>*> ret;

	for(unsigned int i = 0 ; i < loadersRegistry.size() ; i++)
		if(loadersRegistry[i]->Handleable(ad))
			ret.push_back(loadersRegistry[i]);

	return ret;
}

} // namespace Import

} // namespace Algo

} // namespace CGoGN

#include "Algo/Import/AHEMImporterDefAttr.hpp"
