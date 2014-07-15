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

#ifndef _IMPORT2TABLE_H
#define _IMPORT2TABLE_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "Geometry/vector_gen.h"
#include "Geometry/matrix.h"

#include "Utils/gzstream.h"

#include "Algo/Import/importFileTypes.h"
#include "Algo/Modelisation/voxellisation.h"

// #ifdef WITH_ASSIMP
// #include "Assimp/assimp.h"
// #include "Assimp/aiPostProcess.h"
// #include "Assimp/aiScene.h"
// #endif

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{

template <typename PFP>
class MeshTablesSurface
{
public:
    typedef typename PFP::MAP MAP ;
    typedef typename PFP::VEC3 VEC3 ;
    typedef typename VEC3::value_type DATA_TYPE ;
    typedef typename PFP::REAL REAL ;
    enum VOLUME_TYPE {
        TETRAHEDRON = 0,
        HEXAHEDRON,
        SQUARE_PYRAMID,
        TRIANGULAR_PRISM
    };
protected:
    MAP& m_map;

    unsigned int m_nbVertices;

    unsigned int m_nbFaces;

    unsigned int m_lab;
    std::vector<VOLUME_TYPE> m_volumeType;
    /**
    * number of edges per face
    */
    std::vector<unsigned> m_nbEdges;
    std::vector<unsigned>    m_nbVerticesPerVolume;
    /**
    * table of emb indice (for each edge, first vertex)
    */
    std::vector<unsigned int> m_emb;

#ifdef WITH_ASSIMP
    void extractMeshRec(AttributeContainer& container, VertexAttribute<typename PFP::VEC3>& positions, const struct aiScene* scene, const struct aiNode* nd, struct aiMatrix4x4* trafo);
#endif

    bool importTrian(const std::string& filename, std::vector<std::string>& attrNames);

    bool importTrianBinGz(const std::string& filename, std::vector<std::string>& attrNames);

    bool importOff(const std::string& filename, std::vector<std::string>& attrNames);

    bool importMeshBin(const std::string& filename, std::vector<std::string>& attrNames);

    bool importObj(const std::string& filename, std::vector<std::string>& attrNames);

    bool importPly(const std::string& filename, std::vector<std::string>& attrNames);

    //bool importPlyPTM(const std::string& filename, std::vector<std::string>& attrNames);

    bool importPlySLFgeneric(const std::string& filename, std::vector<std::string>& attrNames);

    bool importPlySLFgenericBin(const std::string& filename, std::vector<std::string>& attrNames);

#ifdef WITH_ASSIMP
    bool importASSIMP(const std::string& filename, std::vector<std::string>& attrNames);
#endif	

    bool importAHEM(const std::string& filename, std::vector<std::string>& attrNames);

    bool importSTLAscii(const std::string& filename, std::vector<std::string>& attrNames);

    bool importSTLBin(const std::string& filename, std::vector<std::string>& attrNames);

public:
    //static ImportType getFileType(const std::string& filename);

    bool mergeCloseVertices();

    inline unsigned getNbFaces() const { return m_nbFaces; }

    inline unsigned getNbVertices() const { return m_nbVertices; }

    inline short getNbEdgesFace(int i) const  { return m_nbEdges[i]; }

    inline unsigned int getEmbIdx(int i) { return  m_emb[i]; }

    bool importMesh(const std::string& filename, std::vector<std::string>& attrNames);

    bool importVoxellisation(Algo::Surface::Modelisation::Voxellisation& voxellisation, std::vector<std::string>& attrNames);

    MeshTablesSurface(typename PFP::MAP& map):
        m_map(map)
    { }
};

} // namespace Import

} // namespace Surface


namespace Volume
{
namespace Import
{

template <typename PFP>
class MeshTablesVolume
{
public:
    typedef typename PFP::MAP MAP ;
    typedef typename PFP::VEC3 VEC3 ;
    typedef typename VEC3::value_type DATA_TYPE ;
    typedef typename PFP::REAL REAL ;

    enum VOLUME_TYPE {
        TETRAHEDRON = 0,
        HEXAHEDRON,
        SQUARE_PYRAMID,
        TRIANGULAR_PRISM,
        CONNECTOR
    };

protected:
    MAP& m_map;

    unsigned int m_nbVertices;

    unsigned int m_nbVolumes;
    unsigned int m_nbFaces;
    std::vector<unsigned> m_nbEdges;

    std::vector<unsigned>    m_nbVerticesPerVolume;
    std::vector<VOLUME_TYPE> m_volumeType;

    /**
    * table of emb ptr (for each face, first vertex)
    */
    std::vector<unsigned int> m_emb;

    //Tetrahedra

    /**
     * @brief importTet
     * @param filename
     * @param attrNames
     * @return
     */
    bool importTet(const std::string& filename, std::vector<std::string>& attrNames);

    bool importOFFWithELERegions(const std::string& filenameOFF, const std::string& filenameELE, std::vector<std::string>& attrNames);

    bool importNodeWithELERegions(const std::string& filenameNode, const std::string& filenameELE, std::vector<std::string>& attrNames);

    bool importTetmesh(const std::string& filename, std::vector<std::string>& attrNames);

    bool importTs(const std::string& filename, std::vector<std::string>& attrNames);

    //

    bool importMSH(const std::string& filename, std::vector<std::string>& attrNames);

    //    bool importVTU(const std::string& filename, std::vector<std::string>& attrNames);

    bool importNAS(const std::string& filename, std::vector<std::string>& attrNames);

    //    bool importVBGZ(const std::string& filename, std::vector<std::string>& attrNames);

    //bool importMoka(const std::string& filename, std::vector<std::string>& attrNames);

    //bool importOVM(const std::string& filename, std::vector<std::string>& attrNames);

public:
    //static ImportType getFileType(const std::string& filename);

    inline static unsigned int getNbVerticesOfVolume(VOLUME_TYPE vt) {
        if (vt == CONNECTOR)
            return 4u;
        else if (vt == TETRAHEDRON)
            return 4u;
        else if ( vt == SQUARE_PYRAMID)
            return 5u;
        else if (vt == TRIANGULAR_PRISM)
            return 6u;
        else if ( vt == HEXAHEDRON )
            return 8u;
        return 0xFFFFFF;
    }

    static bool mergeCloseVertices() { return true;}

    inline unsigned getNbVertices() const { return m_nbVertices; }

    inline unsigned getNbVolumes() const { return m_nbVolumes; }

//    inline short getNbFacesVolume(int i) const { return m_nbVerticesPerVolume[i]; }

    inline VOLUME_TYPE getVolumeType(int i) const { return m_volumeType[i] ;}

    inline unsigned int getEmbIdx(int i) const { return  m_emb[i]; }

    bool importMesh(const std::string& filename, std::vector<std::string>& attrNames);

    MeshTablesVolume(typename PFP::MAP& map):
        m_map(map)
    { }
};


} // namespace Import

} // namespace Volume


} // namespace Algo

} // namespace CGoGN

#include "Algo/Import/import2tablesSurface.hpp"
#include "Algo/Import/import2tablesVolume.hpp"

#endif 
