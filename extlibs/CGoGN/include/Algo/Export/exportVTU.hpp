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

#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/autoAttributeHandler.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor2.h"
#include "Topology/generic/cellmarker.h"

#include "Utils/compress.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Export
{

template <typename PFP>
bool exportVTU(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const char* filename)
{
	if (map.dimension() != 2)
	{
		CGoGNerr << "Surface::Export::exportVTU works only with map of dimension 2"<< CGoGNendl;
		return false;
	}

	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

	// open file
	std::ofstream fout ;
	fout.open(filename, std::ios::out) ;

	if (!fout.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
		return false ;
	}

	VertexAutoAttribute<unsigned int> indices(map,"indices_vert");

	unsigned int count=0;
	for (unsigned int i = position.begin(); i != position.end(); position.next(i))
	{
		indices[i] = count++;
	}

	std::vector<unsigned int> triangles;
	std::vector<unsigned int> quads;
	std::vector<unsigned int> others;
	std::vector<unsigned int> others_begin;
	triangles.reserve(2048);
	quads.reserve(2048);
	others.reserve(2048);
	others_begin.reserve(2048);

	TraversorF<MAP> trav(map) ;
	for(Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		unsigned int degree = map.faceDegree(d);
		Dart f=d;
		switch(degree)
		{
			case 3:
				triangles.push_back(indices[f]); f = map.phi1(f);
				triangles.push_back(indices[f]); f = map.phi1(f);
				triangles.push_back(indices[f]);
				break;
			case 4:
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]);
				break;

			default:
				others_begin.push_back(others.size());
				do
				{
					others.push_back(indices[f]); f = map.phi1(f);

				} while (f!=d);
				break;
		}
	}
	others_begin.push_back(others.size());

	unsigned int nbtotal = triangles.size()/3 + quads.size()/4 + others_begin.size()-1;

	fout << "<?xml version=\"1.0\"?>" << std::endl;
	fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
	fout << "<UnstructuredGrid>" <<  std::endl;
	fout << "<Piece NumberOfPoints=\"" << position.nbElements() << "\" NumberOfCells=\""<< nbtotal << "\">" << std::endl;
	fout << "<Points>" << std::endl;
	fout << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;

	for (unsigned int i = position.begin(); i != position.end(); position.next(i))
	{
		const VEC3& P = position[i];
		fout << P[0]<< " " << P[1]<< " " << P[2] << std::endl;
	}

	fout << "</DataArray>" << std::endl;
	fout << "</Points>" << std::endl;
	fout << "<Cells>" << std::endl;
	fout << "<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;

	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		fout << triangles[i]   << " " << triangles[i+1] << " " << triangles[i+2] << std::endl;
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		fout << quads[i]   << " " << quads[i+1] << " " << quads[i+2] << " " << quads[i+3]<< std::endl;
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int beg = others_begin[i-1];
		unsigned int end = others_begin[i];
		for (unsigned int j=beg; j<end; ++j)
		{
			fout <<  others[j] << " ";
		}
		fout << std::endl;
	}

	fout << "</DataArray>" << std::endl;
	fout << "<DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" ;

	unsigned int offset = 0;
	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		offset += 3;
		if (i%60 ==0)
			fout << std::endl;
		fout << " " << offset;
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		offset += 4;
		if (i%80 ==0)
			fout << std::endl;
		fout << " "<< offset;
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int length = others_begin[i] - others_begin[i-1];
		offset += length;
		if (i%20 ==0)
			fout << std::endl;
		fout << " "<< offset;
		fout << std::endl;
	}

	fout << std::endl << "</DataArray>" << std::endl;
	fout << "<DataArray type=\"UInt8\" Name=\"types\" Format=\"ascii\">";
	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		if (i%60 ==0)
			fout << std::endl;
		fout << " 5";
	}
	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		if (i%80 ==0)
			fout << std::endl;
		fout << " 9";
	}
	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		if (i%20 ==0)
			fout << std::endl;
		fout << " 7";
	}

	fout << std::endl << "</DataArray>" << std::endl;
	fout << "</Cells>" << std::endl;
	fout << "</Piece>" << std::endl;
	fout << "</UnstructuredGrid>" << std::endl;
	fout << "</VTKFile>" << std::endl;

	fout.close();
	return true;
}



template <typename PFP>
bool exportVTUBinary(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const char* filename)
{
	if (map.dimension() != 2)
	{
		CGoGNerr << "Surface::Export::exportVTU works only with map of dimension 2"<< CGoGNendl;
		return false;
	}

	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

	// open file
	std::ofstream fout ;
	fout.open(filename, std::ios_base::out | std::ios_base::trunc) ;

	if (!fout.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
		return false ;
	}

	VertexAutoAttribute<unsigned int> indices(map,"indices_vert");

	unsigned int count=0;
	for (unsigned int i = position.begin(); i != position.end(); position.next(i))
	{
		indices[i] = count++;
	}

	std::vector<unsigned int> triangles;
	std::vector<unsigned int> quads;
	std::vector<unsigned int> others;
	std::vector<unsigned int> others_begin;
	triangles.reserve(2048);
	quads.reserve(2048);
	others.reserve(2048);
	others_begin.reserve(2048);

	TraversorF<MAP> trav(map) ;
	for(Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		unsigned int degree = map.faceDegree(d);
		Dart f=d;
		switch(degree)
		{
			case 3:
				triangles.push_back(indices[f]); f = map.phi1(f);
				triangles.push_back(indices[f]); f = map.phi1(f);
				triangles.push_back(indices[f]);
				break;
			case 4:
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]);
				break;

			default:
				others_begin.push_back(others.size());
				do
				{
					others.push_back(indices[f]); f = map.phi1(f);

				} while (f!=d);
				break;
		}
	}
	others_begin.push_back(others.size());

	unsigned int nbtotal = triangles.size()/3 + quads.size()/4 + others_begin.size()-1;

	fout << "<?xml version=\"1.0\"?>" << std::endl;
	fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
	fout << "<UnstructuredGrid>" <<  std::endl;
	fout << "<Piece NumberOfPoints=\"" << position.nbElements() << "\" NumberOfCells=\""<< nbtotal << "\">" << std::endl;
	fout << "<Points>" << std::endl;
	fout << "<DataArray type =\"Float32\" Name =\"Position\" NumberOfComponents =\"3\" format =\"appended\" offset =\"0\"/>"  << std::endl;

	unsigned int offsetAppend = position.nbElements() * 3 * sizeof(float) + sizeof(unsigned int);	// Data + sz of blk

	fout << "</Points>" << std::endl;

	fout << "<Cells>" << std::endl;

	std::vector<int> bufferInt;
	bufferInt.reserve(triangles.size()+quads.size()+others.size());

	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		bufferInt.push_back(triangles[i]);
		bufferInt.push_back(triangles[i+1]);
		bufferInt.push_back(triangles[i+2]);
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		bufferInt.push_back(quads[i]);
		bufferInt.push_back(quads[i+1]);
		bufferInt.push_back(quads[i+2]);
		bufferInt.push_back(quads[i+3]);
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int beg = others_begin[i-1];
		unsigned int end = others_begin[i];
		for (unsigned int j=beg; j<end; ++j)
			bufferInt.push_back(others[j]);
	}

	fout << "<DataArray type =\"Int32\" Name =\"connectivity\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;
	offsetAppend +=bufferInt.size() * sizeof(unsigned int) + sizeof(unsigned int);

	fout << "<DataArray type =\"Int32\" Name =\"offsets\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;
	offsetAppend += (triangles.size()/3 + quads.size()/4 + others_begin.size()-1) * sizeof(unsigned int) + sizeof(unsigned int);

	fout << "<DataArray type =\"UInt8\" Name =\"types\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

	fout << "</Cells>" << std::endl;

	fout << "</Piece>" << std::endl;
	fout << "</UnstructuredGrid>" << std::endl;
	fout << "<AppendedData encoding=\"raw\">" << std::endl << "_";

	fout.close();
	fout.open(filename, std::ios_base::binary | std::ios_base::ate | std::ios_base::app);

	unsigned int lengthBuff=0;
	// bufferize and save position
	{
		std::vector<VEC3> bufferV3;
		bufferV3.reserve(position.nbElements());
		for (unsigned int i = position.begin(); i != position.end(); position.next(i))
			bufferV3.push_back(position[i]);

		lengthBuff = bufferV3.size()*sizeof(VEC3);
		fout.write((char*)&lengthBuff,sizeof(unsigned int));

		fout.write((char*)&bufferV3[0],lengthBuff);
	}

	// save already buffrized indices of primitives
	lengthBuff = bufferInt.size()*sizeof(unsigned int);
	fout.write((char*)&lengthBuff,sizeof(unsigned int));

	fout.write((char*)&(bufferInt[0]),lengthBuff);


	// bufferize and save offsets of primitives
	bufferInt.clear();
	unsigned int offset = 0;
	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		offset += 3;
		bufferInt.push_back(offset);
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		offset += 4;
		bufferInt.push_back(offset);
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int length = others_begin[i] - others_begin[i-1];
		offset += length;
		bufferInt.push_back(offset);
	}

	lengthBuff = bufferInt.size()*sizeof(unsigned int);
	fout.write((char*)&lengthBuff,sizeof(unsigned int));

	fout.write((char*)&(bufferInt[0]), lengthBuff);

	// bufferize and save types of primitives

	std::vector<unsigned char> bufferUC;
	bufferUC.reserve(triangles.size()/3 + quads.size()/4 + others_begin.size());

	for (unsigned int i=0; i<triangles.size(); i+=3)
		bufferUC.push_back((unsigned char)5);

	for (unsigned int i=0; i<quads.size(); i+=4)
		bufferUC.push_back((unsigned char)9);

	for (unsigned int i=1; i<others_begin.size(); ++i)
		bufferUC.push_back((unsigned char)7);

	lengthBuff = bufferUC.size()*sizeof(unsigned char);
	fout.write((char*)&lengthBuff,sizeof(unsigned int));

	fout.write((char*)&(bufferUC[0]), lengthBuff);

	fout.close();
	fout.open(filename, std::ios_base::ate | std::ios_base::app);

	fout << std::endl << "</AppendedData>" << std::endl;
	fout << "</VTKFile>" << std::endl;

	fout.close();
	return true;
}



/*
template <typename PFP>
bool exportVTUCompressed(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const char* filename)
{
	if (map.dimension() != 2)
	{
		CGoGNerr << "Surface::Export::exportVTU works only with map of dimension 2"<< CGoGNendl;
		return false;
	}

	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

	// open file
	std::ofstream fout ;
	fout.open(filename, std::ios_base::out | std::ios_base::trunc) ;

	if (!fout.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
		return false ;
	}

	VertexAutoAttribute<unsigned int> indices(map,"indices_vert");

	unsigned int count=0;
	for (unsigned int i = position.begin(); i != position.end(); position.next(i))
	{
		indices[i] = count++;
	}

	std::vector<unsigned int> triangles;
	std::vector<unsigned int> quads;
	std::vector<unsigned int> others;
	std::vector<unsigned int> others_begin;
	triangles.reserve(2048);
	quads.reserve(2048);
	others.reserve(2048);
	others_begin.reserve(2048);

	TraversorF<MAP> trav(map) ;
	for(Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		unsigned int degree = map.faceDegree(d);
		Dart f=d;
		switch(degree)
		{
			case 3:
				triangles.push_back(indices[f]); f = map.phi1(f);
				triangles.push_back(indices[f]); f = map.phi1(f);
				triangles.push_back(indices[f]);
				break;
			case 4:
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]); f = map.phi1(f);
				quads.push_back(indices[f]);
				break;

			default:
				others_begin.push_back(others.size());
				do
				{
					others.push_back(indices[f]); f = map.phi1(f);

				} while (f!=d);
				break;
		}
	}
	others_begin.push_back(others.size());

	unsigned int nbtotal = triangles.size()/3 + quads.size()/4 + others_begin.size()-1;

	fout << "<?xml version=\"1.0\"?>" << std::endl;
	fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << std::endl;
	fout << "<UnstructuredGrid>" <<  std::endl;
	fout << "<Piece NumberOfPoints=\"" << position.nbElements() << "\" NumberOfCells=\""<< nbtotal << "\">" << std::endl;
	fout << "<Points>" << std::endl;
	fout << "<DataArray type =\"Float32\" Name =\"Position\" NumberOfComponents =\"3\" format =\"appended\" offset =\"0\"/>"  << std::endl;

	unsigned int offsetAppend = position.nbElements() * 3 * sizeof(float) + sizeof(unsigned int);	// Data + sz of blk

	fout << "</Points>" << std::endl;

	fout << "<Cells>" << std::endl;

	std::vector<int> bufferInt;
	bufferInt.reserve(triangles.size()+quads.size()+others.size());

	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		bufferInt.push_back(triangles[i]);
		bufferInt.push_back(triangles[i+1]);
		bufferInt.push_back(triangles[i+2]);
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		bufferInt.push_back(quads[i]);
		bufferInt.push_back(quads[i+1]);
		bufferInt.push_back(quads[i+2]);
		bufferInt.push_back(quads[i+3]);
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int beg = others_begin[i-1];
		unsigned int end = others_begin[i];
		for (unsigned int j=beg; j<end; ++j)
			bufferInt.push_back(others[j]);
	}

	fout << "<DataArray type =\"Int32\" Name =\"connectivity\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;
	offsetAppend +=bufferInt.size() * sizeof(unsigned int) + sizeof(unsigned int);

	fout << "<DataArray type =\"Int32\" Name =\"offsets\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;
	offsetAppend += (triangles.size()/3 + quads.size()/4 + others_begin.size()-1) * sizeof(unsigned int) + sizeof(unsigned int);

	fout << "<DataArray type =\"UInt8\" Name =\"types\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

	fout << "</Cells>" << std::endl;

	fout << "</Piece>" << std::endl;
	fout << "</UnstructuredGrid>" << std::endl;
	fout << "<AppendedData encoding=\"raw\">" << std::endl << "_";

	fout.close();
	fout.open(filename, std::ios_base::binary | std::ios_base::ate | std::ios_base::app);

	unsigned int lengthBuff=0;
	// bufferize and save position
	{
		std::vector<VEC3> bufferV3;
		bufferV3.reserve(position.nbElements());
		for (unsigned int i = position.begin(); i != position.end(); position.next(i))
			bufferV3.push_back(position[i]);

		lengthBuff = bufferV3.size()*sizeof(VEC3);
		fout.write((char*)&lengthBuff,sizeof(unsigned int));

		Utils::zlibVTUWriteCompressed((unsigned char*)&bufferV3[0], lengthBuff, fout);
//		fout.write((char*)&bufferV3[0],lengthBuff);
	}

	// save already buffrized indices of primitives
	lengthBuff = bufferInt.size()*sizeof(unsigned int);
	fout.write((char*)&lengthBuff,sizeof(unsigned int));

	Utils::zlibVTUWriteCompressed((unsigned char*)&(bufferInt[0]), lengthBuff, fout);
//	fout.write((char*)&(bufferInt[0]),lengthBuff);


	// bufferize and save offsets of primitives
	bufferInt.clear();
	unsigned int offset = 0;
	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		offset += 3;
		bufferInt.push_back(offset);
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		offset += 4;
		bufferInt.push_back(offset);
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int length = others_begin[i] - others_begin[i-1];
		offset += length;
		bufferInt.push_back(offset);
	}

	lengthBuff = bufferInt.size()*sizeof(unsigned int);
	fout.write((char*)&lengthBuff,sizeof(unsigned int));

	Utils::zlibVTUWriteCompressed((unsigned char*)&(bufferInt[0]), lengthBuff, fout);
//	fout.write((char*)&(bufferInt[0]), lengthBuff);

	// bufferize and save types of primitives

	std::vector<unsigned char> bufferUC;
	bufferUC.reserve(triangles.size()/3 + quads.size()/4 + others_begin.size());

	for (unsigned int i=0; i<triangles.size(); i+=3)
		bufferUC.push_back((unsigned char)5);

	for (unsigned int i=0; i<quads.size(); i+=4)
		bufferUC.push_back((unsigned char)9);

	for (unsigned int i=1; i<others_begin.size(); ++i)
		bufferUC.push_back((unsigned char)7);

	lengthBuff = bufferUC.size()*sizeof(unsigned char);
	fout.write((char*)&lengthBuff,sizeof(unsigned int));

	Utils::zlibVTUWriteCompressed((unsigned char*)&(bufferUC[0]), lengthBuff, fout);
//	fout.write((char*)&(bufferUC[0]), lengthBuff);


	fout.close();
	fout.open(filename, std::ios_base::ate | std::ios_base::app);

	fout << std::endl << "</AppendedData>" << std::endl;
	fout << "</VTKFile>" << std::endl;

	fout.close();
	return true;
}

*/




// COMPLETE VERSION

template <typename PFP>
VTUExporter<PFP>::VTUExporter(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position):
	m_map(map),m_position(position),
	nbtotal(0),noPointData(true),noCellData(true),closed(false),binaryMode(false),f_tempoBin_out(NULL)
{
	if (map.dimension() != 2)
	{
		CGoGNerr << "Surface::Export::exportVTU works only with map of dimension 2"<< CGoGNendl;
	}

}

template <typename PFP>
bool VTUExporter<PFP>::init(const char* filename, bool bin)
{
	// save filename for close open ?
	m_filename = std::string(filename);

	// open file
	fout.open(filename, std::ios::out) ;

	if (!fout.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
		return false;
	}

	VertexAutoAttribute<unsigned int> indices(m_map,"indices_vert");

	unsigned int count=0;
	for (unsigned int i = m_position.begin(); i != m_position.end(); m_position.next(i))
	{
		indices[i] = count++;
	}

	triangles.reserve(4096);
	quads.reserve(4096);
	others.reserve(4096);
	others_begin.reserve(4096);

	bufferTri.reserve(4096);
	bufferQuad.reserve(4096);
	bufferOther.reserve(4096);

	TraversorF<MAP> trav(m_map) ;
	for(Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		unsigned int degree = m_map.faceDegree(d);
		Dart f=d;
		switch(degree)
		{
			case 3:
				bufferTri.push_back(d);
				triangles.push_back(indices[f]); f = m_map.phi1(f);
				triangles.push_back(indices[f]); f = m_map.phi1(f);
				triangles.push_back(indices[f]);
				break;
			case 4:
				bufferQuad.push_back(d);
				quads.push_back(indices[f]); f = m_map.phi1(f);
				quads.push_back(indices[f]); f = m_map.phi1(f);
				quads.push_back(indices[f]); f = m_map.phi1(f);
				quads.push_back(indices[f]);
				break;

			default:
				bufferOther.push_back(d);
				others_begin.push_back(others.size());
				do
				{
					others.push_back(indices[f]); f = m_map.phi1(f);

				} while (f!=d);
				break;
		}
	}
	others_begin.push_back(others.size());

	nbtotal = triangles.size()/3 + quads.size()/4 + others_begin.size()-1;

	fout << "<?xml version=\"1.0\"?>" << std::endl;
	fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
	fout << "<UnstructuredGrid>" <<  std::endl;
	fout << "<Piece NumberOfPoints=\"" << m_position.nbElements() << "\" NumberOfCells=\""<< nbtotal << "\">" << std::endl;

	if (bin)
	{
		binaryMode = true;
		f_tempoBin_out = tmpfile();
		offsetAppend = 0;
	}

	return true;
}


template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType,  unsigned int nbComp, const std::string& name)
{
	if (binaryMode)
		return addBinaryVertexAttribute(attrib,vtkType,nbComp,name);

	if (!noCellData)
	{
		CGoGNerr << "VTUExporter<PFP>::addVertexAttribute: endFaceAttributes before adding VertexAttribute"<< CGoGNendl;
		return;
	}

	if (noPointData)
	{
		fout << "<PointData Scalars=\"scalars\">" << std::endl;
		noPointData = false;
	}

	if (nbComp==0)
		switch(vtkType[vtkType.size()-1])
		{
			case '2': // 32 (Float32)
				nbComp = sizeof(T)/4;
				break;
			case '4': // 64	(Int64)
				nbComp = sizeof(T)/8;
				break;
			case '8': // 8 (Uint8)
				nbComp = sizeof(T);
				break;
			case '6': // 16 (Int16)
				nbComp = sizeof(T)/2;
				break;
			default:
				break;
		}


	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;

	// assume that std::cout of attribute is "c0 c1 c2 ..."
	for (unsigned int i = attrib.begin(); i != attrib.end(); attrib.next(i))
		fout << attrib[i] << std::endl;

	fout << "</DataArray>" << std::endl;
}



template <typename PFP>
void VTUExporter<PFP>::endVertexAttributes()
{
	if (!noPointData)
		fout << "</PointData>" << std::endl;

	noPointData = true;
}



template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addFaceAttribute(const FaceAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp, const std::string& name)
{
	if (binaryMode)
		return addBinaryFaceAttribute(attrib,vtkType,nbComp,name);


	if (!noPointData)
	{
		CGoGNerr << "VTUExporter<PFP>::addFaceAttribute: endVertexAttributes before adding FaceAttribute"<< CGoGNendl;
		return;
	}

	if (noCellData)
	{
		fout << "<CellData Scalars=\"scalars\">" << std::endl;
		noCellData = false;
	}

	if (nbComp==0)
		switch(vtkType[vtkType.size()-1])
		{
			case '2': // 32 (Float32)
				nbComp = sizeof(T)/4;
				break;
			case '4': // 64	(Int64)
				nbComp = sizeof(T)/8;
				break;
			case '8': // 8 (Uint8)
				nbComp = sizeof(T);
				break;
			case '6': // 16 (Int16)
				nbComp = sizeof(T)/2;
				break;
			default:
				break;
		}


	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;

	// assume that std::cout of attribute is "c0 c1 c2 ..."
	for (typename std::vector<Dart>::iterator it = bufferTri.begin(); it != bufferTri.end(); ++it)
		fout << attrib[*it] << std::endl;
	for (typename std::vector<Dart>::iterator it = bufferQuad.begin(); it != bufferQuad.end(); ++it)
		fout << attrib[*it] << std::endl;
	for (typename std::vector<Dart>::iterator it = bufferOther.begin(); it != bufferOther.end(); ++it)
		fout << attrib[*it] << std::endl;

	fout << "</DataArray>" << std::endl;

}

template <typename PFP>
void VTUExporter<PFP>::endFaceAttributes()
{
	if (!noCellData)
		fout << "</CellData>" << std::endl;

	noCellData = true;
}


template <typename PFP>
bool VTUExporter<PFP>::close()
{
	if (binaryMode)
		return binaryClose();

	if (!noPointData)
		endVertexAttributes();

	if (!noCellData)
		endFaceAttributes();

	fout << "<Points>" << std::endl;
	fout << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;

	for (unsigned int i = m_position.begin(); i != m_position.end(); m_position.next(i))
	{
		const VEC3& P = m_position[i];
		fout << P[0]<< " " << P[1]<< " " << P[2] << std::endl;
	}

	fout << "</DataArray>" << std::endl;
	fout << "</Points>" << std::endl;
	fout << "<Cells>" << std::endl;
	fout << "<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;

	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		fout << triangles[i]   << " " << triangles[i+1] << " " << triangles[i+2] << std::endl;
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		fout << quads[i]   << " " << quads[i+1] << " " << quads[i+2] << " " << quads[i+3]<< std::endl;
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int beg = others_begin[i-1];
		unsigned int end = others_begin[i];
		for (unsigned int j=beg; j<end; ++j)
		{
			fout <<  others[j] << " ";
		}
		fout << std::endl;
	}

	fout << "</DataArray>" << std::endl;
	fout << "<DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" ;

	unsigned int offset = 0;
	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		offset += 3;
		if (i%60 ==0)
			fout << std::endl;
		fout << " " << offset;
	}

	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		offset += 4;
		if (i%80 ==0)
			fout << std::endl;
		fout << " "<< offset;
	}

	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		unsigned int length = others_begin[i] - others_begin[i-1];
		offset += length;
		if (i%20 ==0)
			fout << std::endl;
		fout << " "<< offset;
	}

	fout << std::endl << "</DataArray>" << std::endl;
	fout << "<DataArray type=\"UInt8\" Name=\"types\" Format=\"ascii\">";
	for (unsigned int i=0; i<triangles.size(); i+=3)
	{
		if (i%60 ==0)
			fout << std::endl;
		fout << " 5";
	}
	for (unsigned int i=0; i<quads.size(); i+=4)
	{
		if (i%80 ==0)
			fout << std::endl;
		fout << " 9";
	}
	for (unsigned int i=1; i<others_begin.size(); ++i)
	{
		if (i%20 ==0)
			fout << std::endl;
		fout << " 7";
	}

	fout << std::endl << "</DataArray>" << std::endl;
	fout << "</Cells>" << std::endl;
	fout << "</Piece>" << std::endl;
	fout << "</UnstructuredGrid>" << std::endl;
	fout << "</VTKFile>" << std::endl;

	fout.close();
	closed=true;
	return true;
}



// BINARY FUCNTION


template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addBinaryVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType,  unsigned int nbComp, const std::string& name)
{
	if (!noCellData)
	{
		CGoGNerr << "VTUExporter<PFP>::addVertexAttribute: endFaceAttributes before adding VertexAttribute"<< CGoGNendl;
		return;
	}

	if (noPointData)
	{
		fout << "<PointData Scalars=\"scalars\">" << std::endl;
		noPointData = false;
	}

	if (nbComp==0)
		switch(vtkType[vtkType.size()-1])
		{
			case '2': // 32 (Float32)
				nbComp = sizeof(T)/4;
				break;
			case '4': // 64	(Int64)
				nbComp = sizeof(T)/8;
				break;
			case '8': // 8 (Uint8)
				nbComp = sizeof(T);
				break;
			case '6': // 16 (Int16)
				nbComp = sizeof(T)/2;
				break;
			default:
				break;
		}

	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;

	std::vector<T> buffer;
	buffer.reserve(attrib.nbElements());

	for (unsigned int i = attrib.begin(); i != attrib.end(); attrib.next(i))
		buffer.push_back(attrib[i]);

	unsigned int sz = buffer.size()*sizeof(T);

	fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
	fwrite(&buffer[0], sizeof(T), buffer.size(), f_tempoBin_out);	// block

	offsetAppend += sizeof(T) * buffer.size() + sizeof(unsigned int);
}


template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addBinaryFaceAttribute(const FaceAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp, const std::string& name)
{
	if (!noPointData)
	{
		CGoGNerr << "VTUExporter<PFP>::addFaceAttribute: endVertexAttributes before adding FaceAttribute"<< CGoGNendl;
		return;
	}

	if (noCellData)
	{
		fout << "<CellData Scalars=\"scalars\">" << std::endl;
		noCellData = false;
	}

	if (nbComp==0)
	switch(vtkType[vtkType.size()-1])
	{
		case '2': // 32
			nbComp = sizeof(T)/4;
			break;
		case '4': // 64
			nbComp = sizeof(T)/8;
			break;
		case '8': // 8
			nbComp = sizeof(T);
			break;
		case '6': // 16
			nbComp = sizeof(T)/2;
			break;
	}

	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;

	std::vector<T> buffer;
	buffer.reserve(bufferTri.size() + bufferQuad.size() + bufferOther.size());

	for (typename std::vector<Dart>::iterator it = bufferTri.begin(); it != bufferTri.end(); ++it)
		buffer.push_back(attrib[*it]);
	for (typename std::vector<Dart>::iterator it = bufferQuad.begin(); it != bufferQuad.end(); ++it)
		buffer.push_back(attrib[*it]);
	for (typename std::vector<Dart>::iterator it = bufferOther.begin(); it != bufferOther.end(); ++it)
		buffer.push_back(attrib[*it]);

	unsigned int sz = buffer.size()*sizeof(T);
	fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
	fwrite(&buffer[0], sizeof(T), buffer.size(), f_tempoBin_out);	// block

	offsetAppend += sizeof(T) * buffer.size() + sizeof(unsigned int);
}


template <typename PFP>
bool VTUExporter<PFP>::binaryClose()
{
	if (!noPointData)
		endVertexAttributes();

	if (!noCellData)
		endFaceAttributes();

	{ // just for scope of std::vector (free memory)
		fout << "<Points>" << std::endl;
		fout << "<DataArray type =\"Float32\" Name =\"Position\" NumberOfComponents =\"3\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

		std::vector<VEC3> buffer;
		buffer.reserve(m_position.nbElements());

		for (unsigned int i = m_position.begin(); i != m_position.end(); m_position.next(i))
			buffer.push_back(m_position[i]);

		unsigned int sz = buffer.size()*sizeof(VEC3);

		offsetAppend += sz + sizeof(unsigned int);

		fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
		fwrite(&buffer[0], sizeof(VEC3), buffer.size(), f_tempoBin_out);	// block

		fout << "</Points>" << std::endl;
	}

	{ // just for scope of std::vector (free memory)

		fout << "<Cells>" << std::endl;
		fout << "<DataArray type =\"Int32\" Name =\"connectivity\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

		std::vector<int> bufferInt;
		bufferInt.reserve(triangles.size()+quads.size()+others.size());

		for (unsigned int i=0; i<triangles.size(); i+=3)
		{
			bufferInt.push_back(triangles[i]);
			bufferInt.push_back(triangles[i+1]);
			bufferInt.push_back(triangles[i+2]);
		}

		for (unsigned int i=0; i<quads.size(); i+=4)
		{
			bufferInt.push_back(quads[i]);
			bufferInt.push_back(quads[i+1]);
			bufferInt.push_back(quads[i+2]);
			bufferInt.push_back(quads[i+3]);
		}

		for (unsigned int i=1; i<others_begin.size(); ++i)
		{
			unsigned int beg = others_begin[i-1];
			unsigned int end = others_begin[i];
			for (unsigned int j=beg; j<end; ++j)
				bufferInt.push_back(others[j]);
		}

		unsigned int sz =  bufferInt.size() * sizeof(unsigned int);
		offsetAppend += sz + sizeof(unsigned int);
		fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
		fwrite(&bufferInt[0], sizeof(unsigned int), bufferInt.size(), f_tempoBin_out);	// block

		fout << "<DataArray type =\"Int32\" Name =\"offsets\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

		bufferInt.clear();
		unsigned int offset = 0;
		for (unsigned int i=0; i<triangles.size(); i+=3)
		{
			offset += 3;
			bufferInt.push_back(offset);
		}

		for (unsigned int i=0; i<quads.size(); i+=4)
		{
			offset += 4;
			bufferInt.push_back(offset);
		}

		for (unsigned int i=1; i<others_begin.size(); ++i)
		{
			unsigned int length = others_begin[i] - others_begin[i-1];
			offset += length;
			bufferInt.push_back(offset);
		}

		sz =  bufferInt.size() * sizeof(unsigned int);

		offsetAppend += sz + sizeof(unsigned int);
		fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
		fwrite(&bufferInt[0], sizeof(unsigned int), bufferInt.size(), f_tempoBin_out);	// block

	}

	fout << "<DataArray type =\"UInt8\" Name =\"types\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;
	std::vector<unsigned char> bufferUC;
	bufferUC.reserve(triangles.size()/3 + quads.size()/4 + others_begin.size());

	for (unsigned int i=0; i<triangles.size(); i+=3)
		bufferUC.push_back((unsigned char)5);

	for (unsigned int i=0; i<quads.size(); i+=4)
		bufferUC.push_back((unsigned char)9);

	for (unsigned int i=1; i<others_begin.size(); ++i)
		bufferUC.push_back((unsigned char)7);

	unsigned int sz =  bufferUC.size() * sizeof(unsigned char);

	offsetAppend += sz + sizeof(unsigned int);

	fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
	fwrite(&bufferUC[0], sizeof(unsigned char), bufferUC.size(), f_tempoBin_out);	// block

	fout << "</Cells>" << std::endl;
	fout << "</Piece>" << std::endl;
	fout << "</UnstructuredGrid>" << std::endl;
	fout << "<AppendedData encoding=\"raw\">" << std::endl << "_";

	fout.close();
	fout.open(m_filename.c_str(), std::ios_base::binary | std::ios_base::ate | std::ios_base::app);

	// copy data from tempo file to final file !

	long int binLength = ftell(f_tempoBin_out);
	rewind (f_tempoBin_out);

	const unsigned int blksz = 1024*1024;
	char* buffer = new char[blksz];

	while (binLength > blksz)
	{
		fread ( buffer, 1, blksz, f_tempoBin_out);
		fout.write((char*)buffer,blksz);
		binLength -= blksz;
	}

	fread ( buffer, 1, binLength, f_tempoBin_out);
	fout.write((char*)buffer,binLength);

	delete[] buffer;
	fclose(f_tempoBin_out);

	fout.close();
	fout.open(m_filename.c_str(), std::ios_base::ate | std::ios_base::app);
	fout << std::endl << "</AppendedData>" << std::endl;
	fout << "</VTKFile>" << std::endl;

	fout.close();
	closed=true;
	return true;
}




template <typename PFP>
VTUExporter<PFP>::~VTUExporter()
{
	if (!closed)
		close();
	closed = true;
}


} // namespace Export

} // namespace Surface



namespace Volume
{
namespace Export
{

template <typename PFP>
VTUExporter<PFP>::VTUExporter(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position):
	m_map(map),m_position(position),
	nbtotal(0),noPointData(true),noCellData(true),closed(false),binaryMode(false),f_tempoBin_out(NULL)
{
	if (map.dimension() != 3)
	{
		CGoGNerr << "Volume::Export::exportVTU works only with map of dimension 3"<< CGoGNendl;
	}

}

template <typename PFP>
bool VTUExporter<PFP>::init(const char* filename, bool bin)
{
	// save filename for close open ?
	m_filename = std::string(filename);

	// open file
	fout.open(filename, std::ios::out) ;

	if (!fout.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
		return false;
	}

	VertexAutoAttribute<unsigned int> indices(m_map,"indices_vert");

	unsigned int count=0;
	for (unsigned int i = m_position.begin(); i != m_position.end(); m_position.next(i))
	{
		indices[i] = count++;
	}

	tetras.reserve(4096);
	hexas.reserve(4096);

	bufferTetra.reserve(4096);
	bufferHexa.reserve(4096);

	TraversorW<MAP> trav(m_map) ;
	for(Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		unsigned int degree = 0 ;

		Traversor3WV<typename PFP::MAP> twv(m_map, d) ;
		for(Dart it = twv.begin(); it != twv.end(); it = twv.next())
		{
			degree++;
		}

		if (degree == 8)
		{
			//CAS HEXAEDRIQUE (ordre 2 quad superposes, le premier en CW)
			bufferHexa.push_back(d);
			Dart e = d;
			Dart f = m_map.template phi<21121>(d);
			hexas.push_back(indices[f]);
			f = m_map.phi_1(f);
			hexas.push_back(indices[f]);
			f = m_map.phi_1(f);
			hexas.push_back(indices[f]);
			f = m_map.phi_1(f);
			hexas.push_back(indices[f]);
			hexas.push_back(indices[e]);
			e = m_map.phi1(e);
			hexas.push_back(indices[e]);
			e = m_map.phi1(e);
			hexas.push_back(indices[e]);
			e = m_map.phi1(e);
			hexas.push_back(indices[e]);
		}
		if (degree == 4)
		{
			//CAS TETRAEDRIQUE
			bufferTetra.push_back(d);
			Dart e = d;
			tetras.push_back(indices[e]);
			e = m_map.phi1(e);
			tetras.push_back(indices[e]);
			e = m_map.phi1(e);
			tetras.push_back(indices[e]);
			e = m_map.template phi<211>(e);
			tetras.push_back(indices[e]);
		}
	}

	nbtotal = tetras.size()/4 + hexas.size()/8;

	fout << "<?xml version=\"1.0\"?>" << std::endl;
	fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
	fout << "<UnstructuredGrid>" <<  std::endl;
	fout << "<Piece NumberOfPoints=\"" << m_position.nbElements() << "\" NumberOfCells=\""<< nbtotal << "\">" << std::endl;

	if (bin)
	{
		binaryMode = true;
		f_tempoBin_out = tmpfile();
		offsetAppend = 0;
	}

	return true;
}


template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType,  unsigned int nbComp, const std::string& name)
{
	if (binaryMode)
		return addBinaryVertexAttribute(attrib,vtkType,nbComp,name);

	if (!noCellData)
	{
		CGoGNerr << "VTUExporter<PFP>::addVertexAttribute: endFaceAttributes before adding VertexAttribute"<< CGoGNendl;
		return;
	}

	if (noPointData)
	{
		fout << "<PointData Scalars=\"scalars\">" << std::endl;
		noPointData = false;
	}

	if (nbComp==0)
		switch(vtkType[vtkType.size()-1])
		{
			case '2': // 32 (Float32)
				nbComp = sizeof(T)/4;
				break;
			case '4': // 64	(Int64)
				nbComp = sizeof(T)/8;
				break;
			case '8': // 8 (Uint8)
				nbComp = sizeof(T);
				break;
			case '6': // 16 (Int16)
				nbComp = sizeof(T)/2;
				break;
			default:
				break;
		}


	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;

	// assume that std::cout of attribute is "c0 c1 c2 ..."
	for (unsigned int i = attrib.begin(); i != attrib.end(); attrib.next(i))
		fout << attrib[i] << std::endl;

	fout << "</DataArray>" << std::endl;
}



template <typename PFP>
void VTUExporter<PFP>::endVertexAttributes()
{
	if (!noPointData)
		fout << "</PointData>" << std::endl;

	noPointData = true;
}



template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addVolumeAttribute(const VolumeAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp, const std::string& name)
{
	if (binaryMode)
		return addBinaryVolumeAttribute(attrib,vtkType,nbComp,name);


	if (!noPointData)
	{
		CGoGNerr << "VTUExporter<PFP>::addVolumeAttribute: endVertexAttributes before adding FaceAttribute"<< CGoGNendl;
		return;
	}

	if (noCellData)
	{
		fout << "<CellData Scalars=\"scalars\">" << std::endl;
		noCellData = false;
	}

	if (nbComp==0)
		switch(vtkType[vtkType.size()-1])
		{
			case '2': // 32 (Float32)
				nbComp = sizeof(T)/4;
				break;
			case '4': // 64	(Int64)
				nbComp = sizeof(T)/8;
				break;
			case '8': // 8 (Uint8)
				nbComp = sizeof(T);
				break;
			case '6': // 16 (Int16)
				nbComp = sizeof(T)/2;
				break;
			default:
				break;
		}


	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"ascii\">" << std::endl;

	// assume that std::cout of attribute is "c0 c1 c2 ..."
	for (typename std::vector<Dart>::iterator it = bufferTetra.begin(); it != bufferTetra.end(); ++it)
		fout << attrib[*it] << std::endl;
	for (typename std::vector<Dart>::iterator it = bufferHexa.begin(); it != bufferHexa.end(); ++it)
		fout << attrib[*it] << std::endl;

	fout << "</DataArray>" << std::endl;

}

template <typename PFP>
void VTUExporter<PFP>::endVolumeAttributes()
{
	if (!noCellData)
		fout << "</CellData>" << std::endl;

	noCellData = true;
}


template <typename PFP>
bool VTUExporter<PFP>::close()
{
	if (binaryMode)
		return binaryClose();

	if (!noPointData)
		endVertexAttributes();

	if (!noCellData)
		endVolumeAttributes();

	fout << "<Points>" << std::endl;
	fout << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;

	for (unsigned int i = m_position.begin(); i != m_position.end(); m_position.next(i))
	{
		const VEC3& P = m_position[i];
		fout << P[0]<< " " << P[1]<< " " << P[2] << std::endl;
	}

	fout << "</DataArray>" << std::endl;
	fout << "</Points>" << std::endl;
	fout << "<Cells>" << std::endl;
	fout << "<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;

	for (unsigned int i=0; i<tetras.size(); i+=3)
	{
		fout << tetras[i]   << " " << tetras[i+1] << " " << tetras[i+2] << " " << tetras[i+3] << std::endl;
	}

	for (unsigned int i=0; i<hexas.size(); i+=4)
	{
		fout << hexas[i]   << " " << hexas[i+1] << " " << hexas[i+2] << " " << hexas[i+3];
		fout << hexas[i+4] << " " << hexas[i+5] << " " << hexas[i+6] << " " << hexas[i+7]<< std::endl;
	}

	fout << "</DataArray>" << std::endl;
	fout << "<DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" ;

	unsigned int offset = 0;
	for (unsigned int i=0; i<tetras.size(); i+=3)
	{
		offset += 4;
		if (i%60 ==0)
			fout << std::endl;
		fout << " " << offset;
	}

	for (unsigned int i=0; i<hexas.size(); i+=4)
	{
		offset += 8;
		if (i%80 ==0)
			fout << std::endl;
		fout << " "<< offset;
	}


	fout << std::endl << "</DataArray>" << std::endl;
	fout << "<DataArray type=\"UInt8\" Name=\"types\" Format=\"ascii\">";
	for (unsigned int i=0; i<tetras.size(); i+=3)
	{
		if (i%60 ==0)
			fout << std::endl;
		fout << " 10";
	}
	for (unsigned int i=0; i<hexas.size(); i+=4)
	{
		if (i%80 ==0)
			fout << std::endl;
		fout << " 12";
	}
	fout << std::endl << "</DataArray>" << std::endl;
	fout << "</Cells>" << std::endl;
	fout << "</Piece>" << std::endl;
	fout << "</UnstructuredGrid>" << std::endl;
	fout << "</VTKFile>" << std::endl;

	fout.close();
	closed=true;
	return true;
}



// BINARY FUCNTION

template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addBinaryVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType,  unsigned int nbComp, const std::string& name)
{
	if (!noCellData)
	{
		CGoGNerr << "VTUExporter<PFP>::addVertexAttribute: endFaceAttributes before adding VertexAttribute"<< CGoGNendl;
		return;
	}

	if (noPointData)
	{
		fout << "<PointData Scalars=\"scalars\">" << std::endl;
		noPointData = false;
	}

	if (nbComp==0)
		switch(vtkType[vtkType.size()-1])
		{
			case '2': // 32 (Float32)
				nbComp = sizeof(T)/4;
				break;
			case '4': // 64	(Int64)
				nbComp = sizeof(T)/8;
				break;
			case '8': // 8 (Uint8)
				nbComp = sizeof(T);
				break;
			case '6': // 16 (Int16)
				nbComp = sizeof(T)/2;
				break;
			default:
				break;
		}

	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;

	std::vector<T> buffer;
	buffer.reserve(attrib.nbElements());

	for (unsigned int i = attrib.begin(); i != attrib.end(); attrib.next(i))
		buffer.push_back(attrib[i]);

	unsigned int sz = buffer.size()*sizeof(T);

	fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
	fwrite(&buffer[0], sizeof(T), buffer.size(), f_tempoBin_out);	// block

	offsetAppend += sizeof(T) * buffer.size() + sizeof(unsigned int);
}


template <typename PFP>
template<typename T>
void VTUExporter<PFP>::addBinaryVolumeAttribute(const VolumeAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp, const std::string& name)
{
	if (!noPointData)
	{
		CGoGNerr << "VTUExporter<PFP>::addFaceAttribute: endVertexAttributes before adding FaceAttribute"<< CGoGNendl;
		return;
	}

	if (noCellData)
	{
		fout << "<CellData Scalars=\"scalars\">" << std::endl;
		noCellData = false;
	}

	if (nbComp==0)
	switch(vtkType[vtkType.size()-1])
	{
		case '2': // 32
			nbComp = sizeof(T)/4;
			break;
		case '4': // 64
			nbComp = sizeof(T)/8;
			break;
		case '8': // 8
			nbComp = sizeof(T);
			break;
		case '6': // 16
			nbComp = sizeof(T)/2;
			break;
	}

	if (name.size() != 0)
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<name<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;
	else
		fout << "<DataArray type=\""<< vtkType <<"\" Name=\""<<attrib.name()<<"\" NumberOfComponents=\""<< nbComp <<"\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>" << std::endl;

	std::vector<T> buffer;
	buffer.reserve(bufferTetra.size() + bufferHexa.size());

	for (typename std::vector<Dart>::iterator it = bufferTetra.begin(); it != bufferTetra.end(); ++it)
		buffer.push_back(attrib[*it]);
	for (typename std::vector<Dart>::iterator it = bufferHexa.begin(); it != bufferHexa.end(); ++it)
		buffer.push_back(attrib[*it]);


	unsigned int sz = buffer.size()*sizeof(T);
	fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
	fwrite(&buffer[0], sizeof(T), buffer.size(), f_tempoBin_out);	// block

	offsetAppend += sizeof(T) * buffer.size() + sizeof(unsigned int);
}


template <typename PFP>
bool VTUExporter<PFP>::binaryClose()
{
	if (!noPointData)
		endVertexAttributes();

	if (!noCellData)
		endVolumeAttributes();

	{ // just for scope of std::vector (free memory)
		fout << "<Points>" << std::endl;
		fout << "<DataArray type =\"Float32\" Name =\"Position\" NumberOfComponents =\"3\" Format=\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

		std::vector<VEC3> buffer;
		buffer.reserve(m_position.nbElements());

		for (unsigned int i = m_position.begin(); i != m_position.end(); m_position.next(i))
			buffer.push_back(m_position[i]);

		unsigned int sz = buffer.size()*sizeof(VEC3);

		offsetAppend += sz + sizeof(unsigned int);

		fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
		fwrite(&buffer[0], sizeof(VEC3), buffer.size(), f_tempoBin_out);	// block

		fout << "</Points>" << std::endl;
	}

	{ // just for scope of std::vector (free memory)

		fout << "<Cells>" << std::endl;
		fout << "<DataArray type =\"Int32\" Name =\"connectivity\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

		std::vector<int> bufferInt;
		bufferInt.reserve(tetras.size()+hexas.size());

		for (unsigned int i=0; i<tetras.size(); i+=3)
		{
			bufferInt.push_back(tetras[i]);
			bufferInt.push_back(tetras[i+1]);
			bufferInt.push_back(tetras[i+2]);
			bufferInt.push_back(tetras[i+3]);
		}

		for (unsigned int i=0; i<hexas.size(); i+=4)
		{
			bufferInt.push_back(hexas[i]);
			bufferInt.push_back(hexas[i+1]);
			bufferInt.push_back(hexas[i+2]);
			bufferInt.push_back(hexas[i+3]);
		}

		unsigned int sz =  bufferInt.size() * sizeof(unsigned int);
		offsetAppend += sz + sizeof(unsigned int);
		fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
		fwrite(&bufferInt[0], sizeof(unsigned int), bufferInt.size(), f_tempoBin_out);	// block

		fout << "<DataArray type =\"Int32\" Name =\"offsets\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;

		bufferInt.clear();
		unsigned int offset = 0;
		for (unsigned int i=0; i<tetras.size(); i+=3)
		{
			offset += 4;
			bufferInt.push_back(offset);
		}

		for (unsigned int i=0; i<hexas.size(); i+=4)
		{
			offset += 8;
			bufferInt.push_back(offset);
		}

		sz =  bufferInt.size() * sizeof(unsigned int);

		offsetAppend += sz + sizeof(unsigned int);
		fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
		fwrite(&bufferInt[0], sizeof(unsigned int), bufferInt.size(), f_tempoBin_out);	// block

	}

	fout << "<DataArray type =\"UInt8\" Name =\"types\" format =\"appended\" offset =\""<<offsetAppend<<"\"/>"  << std::endl;
	std::vector<unsigned char> bufferUC;
	bufferUC.reserve(tetras.size()/3 + hexas.size()/4);

	for (unsigned int i=0; i<tetras.size(); i+=4)
		bufferUC.push_back((unsigned char)10);

	for (unsigned int i=0; i<hexas.size(); i+=8)
		bufferUC.push_back((unsigned char)12);


	unsigned int sz =  bufferUC.size() * sizeof(unsigned char);

	offsetAppend += sz + sizeof(unsigned int);

	fwrite(&sz, sizeof(unsigned int), 1, f_tempoBin_out);			// size of block
	fwrite(&bufferUC[0], sizeof(unsigned char), bufferUC.size(), f_tempoBin_out);	// block

	fout << "</Cells>" << std::endl;
	fout << "</Piece>" << std::endl;
	fout << "</UnstructuredGrid>" << std::endl;
	fout << "<AppendedData encoding=\"raw\">" << std::endl << "_";

	fout.close();
	fout.open(m_filename.c_str(), std::ios_base::binary | std::ios_base::ate | std::ios_base::app);

	// copy data from tempo file to final file !

	long int binLength = ftell(f_tempoBin_out);
	rewind (f_tempoBin_out);

	const unsigned int blksz = 1024*1024;
	char* buffer = new char[blksz];

	while (binLength > blksz)
	{
		fread ( buffer, 1, blksz, f_tempoBin_out);
		fout.write((char*)buffer,blksz);
		binLength -= blksz;
	}

	fread ( buffer, 1, binLength, f_tempoBin_out);
	fout.write((char*)buffer,binLength);

	delete[] buffer;
	fclose(f_tempoBin_out);

	fout.close();
	fout.open(m_filename.c_str(), std::ios_base::ate | std::ios_base::app);
	fout << std::endl << "</AppendedData>" << std::endl;
	fout << "</VTKFile>" << std::endl;

	fout.close();
	closed=true;
	return true;
}




template <typename PFP>
VTUExporter<PFP>::~VTUExporter()
{
	if (!closed)
		close();
	closed = true;
}





}
}


} // namespace Algo

} // namespace CGoGN
