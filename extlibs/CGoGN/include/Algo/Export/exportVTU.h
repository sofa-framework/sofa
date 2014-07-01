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

#ifndef __EXPORT_VTU_H__
#define __EXPORT_VTU_H__

#include "Topology/generic/attributeHandler.h"
#include "Algo/Import/importFileTypes.h"


#include <stdint.h>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Export
{


/**
* simple export of the geometry of map into a VTU file (VTK unstructured grid xml format)
* @param map map to be exported
* @param position the position container
* @param filename filename of ply file
* @return true if ok
*/
template <typename PFP>
bool exportVTU(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const char* filename);

/**
* simple export of the geometry of map into a binary VTU file (VTK unstructured grid xml format)
* @param map map to be exported
* @param position the position container
* @param filename filename of ply file
* @return true if ok
*/
template <typename PFP>
bool exportVTUBinary(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const char* filename);

//template <typename PFP>
//bool exportVTUCompressed(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const char* filename);

/**
 * class that allow the export of VTU file (ascii or binary)
 * with vertex and face attributes
 */
template <typename PFP>
class VTUExporter
{
protected:
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

	typename PFP::MAP& m_map;
	const VertexAttribute<typename PFP::VEC3>& m_position;

	unsigned int nbtotal;
	bool noPointData;
	bool noCellData;
	bool closed;

	std::ofstream fout ;

	std::vector<unsigned int> triangles;
	std::vector<unsigned int> quads;
	std::vector<unsigned int> others;
	std::vector<unsigned int> others_begin;

	std::vector<Dart> bufferTri;
	std::vector<Dart> bufferQuad;
	std::vector<Dart> bufferOther;

	std::string m_filename;
	bool binaryMode;
	unsigned int offsetAppend;

	FILE* f_tempoBin_out ;


	template<typename T>
	void addBinaryVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");

	template<typename T>
	void addBinaryFaceAttribute(const FaceAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");

	bool binaryClose();

public:

	VTUExporter(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position);

	~VTUExporter();

	/**
	 * @brief start writing header of vtu file
	 * @param filename
	 * @param bin true if binray mode wanted
	 * @return true if ok
	 */
	bool init(const char* filename, bool bin=false);



	/**
	 * @brief add a vertex attribute
	 * @param attrib vertex attribute
	 * @param vtkType Float32/Int32/..
	 * @param nbComp number of components in attribute (if none computed from size of attribute divide bye the vtkType)
	 * @param name data name, if none then used attribute's name
	 */
	template<typename T>
	void addVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");


	/**
	 * @brief finish adding vertex attributes data
	 */
	void endVertexAttributes();

	/**
	 * @brief add a face attribute
	 * @param attrib vertex attribute
	 * @param vtkType Float32/Int32/..
	 * @param nbComp number of components in attribute (if none computed from size of attribute divide bye the vtkType)
	 * @param name data name, if none then used attribute's name
	 */
	template<typename T>
	void addFaceAttribute(const FaceAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");


	/**
	 * @brief finish adding face attributes data
	 */
	void endFaceAttributes();


	/**
	 * @brief finalize file writing & close (automatically called at destruction of not yet done)
	 * @return true if ok
	 */
	bool close();

};

} // namespace Export

} // Surface

namespace Volume
{
namespace Export
{

/**
 * class that allow the export of VTU file (ascii or binary)
 * with vertex and volume attributes
 */
template <typename PFP>
class VTUExporter
{
protected:
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

	typename PFP::MAP& m_map;
	const VertexAttribute<typename PFP::VEC3>& m_position;

	unsigned int nbtotal;
	bool noPointData;
	bool noCellData;
	bool closed;

	std::ofstream fout ;

	std::vector<unsigned int> tetras;
	std::vector<unsigned int> hexas;

	std::vector<Dart> bufferTetra;
	std::vector<Dart> bufferHexa;

	std::string m_filename;
	bool binaryMode;
	unsigned int offsetAppend;

	FILE* f_tempoBin_out ;

	template<typename T>
	void addBinaryVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");

	template<typename T>
	void addBinaryVolumeAttribute(const VolumeAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");

	bool binaryClose();

public:

	VTUExporter(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position);

	~VTUExporter();

	/**
	 * @brief start writing header of vtu file
	 * @param filename
	 * @param bin true if binray mode wanted
	 * @return true if ok
	 */
	bool init(const char* filename, bool bin=false);



	/**
	 * @brief add a vertex attribute
	 * @param attrib vertex attribute
	 * @param vtkType Float32/Int32/..
	 * @param nbComp number of components in attribute (if none computed from size of attribute divide bye the vtkType)
	 * @param name data name, if none then used attribute's name
	 */
	template<typename T>
	void addVertexAttribute(const VertexAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");


	/**
	 * @brief finish adding vertex attributes data
	 */
	void endVertexAttributes();

	/**
	 * @brief add a volume attribute
	 * @param attrib vertex attribute
	 * @param vtkType Float32/Int32/..
	 * @param nbComp number of components in attribute (if none computed from size of attribute divide bye the vtkType)
	 * @param name data name, if none then used attribute's name
	 */
	template<typename T>
	void addVolumeAttribute(const VolumeAttribute<T>& attrib, const std::string& vtkType, unsigned int nbComp=0, const std::string& name="");


	/**
	 * @brief finish adding volume attributes data
	 */
	void endVolumeAttributes();


	/**
	 * @brief finalize file writing & close (automatically called at destruction of not yet done)
	 * @return true if ok
	 */
	bool close();

};

} // namespace Export

} // Volume





} // namespace Algo

} // namespace CGoGN

#include "Algo/Export/exportVTU.hpp"

#endif
