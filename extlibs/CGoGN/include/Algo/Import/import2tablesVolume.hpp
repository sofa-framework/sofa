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

#include "Geometry/orientation.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Import
{

template <typename PFP>
bool MeshTablesVolume<PFP>::importMesh(const std::string& filename, std::vector<std::string>& attrNames)
{
    ImportType kind = getFileType(filename);

    attrNames.clear() ;

    switch (kind)
    {
    case TET:
        return importTet(filename, attrNames);
        break;
    case OFF:
    {
        size_t pos = filename.rfind(".");
        std::string fileEle = filename;
        fileEle.erase(pos);
        fileEle.append(".ele");
        return importOFFWithELERegions(filename, fileEle, attrNames);
        break;
    }
    case NODE:
    {
        size_t pos = filename.rfind(".");
        std::string fileEle = filename;
        fileEle.erase(pos);
        fileEle.append(".ele");
        return importNodeWithELERegions(filename, fileEle, attrNames);
        break;
    }
    case TETMESH:
        return importTetmesh(filename, attrNames);
        break;
    case TS:
        return importTs(filename, attrNames);
        break;
    case MSH:
        //return importMSH(filename, attrNames);
        break;
    case VTU:
        //return importVTU(filename, attrNames);
        break;
    case NAS:
        //return importNAS(filename, attrNames);
        break;
    case VBGZ:
        //return importVBGZ>(filename, attrNames);
        break;
//	case ImportVolumique::MOKA:
//		return importMoka(filename,attrNames);
//		break;
//	case OVM:
//		return importOVM(filename, attrNames);
//		break;
    default:
        CGoGNerr << "Not yet supported" << CGoGNendl;
        break;
    }
    return false;
}

template <typename PFP>
bool MeshTablesVolume<PFP>::importTet(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3> position =  m_map.template getAttribute<VEC3, VERTEX>("position") ;

    if (!position.isValid())
        position = m_map.template addAttribute<VEC3, VERTEX>("position") ;

    attrNames.push_back(position.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    //open file
    std::ifstream fp(filename.c_str(), std::ios::in);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    std::string ligne;

    // reading number of vertices
    std::getline (fp, ligne);
    std::stringstream oss(ligne);
    oss >> m_nbVertices;

    // reading number of tetrahedra
    std::getline (fp, ligne);
    std::stringstream oss2(ligne);
    oss2 >> m_nbVolumes;

    //reading vertices
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices);

    for(unsigned int i = 0; i < m_nbVertices; ++i)
    {
        do
        {
            std::getline (fp, ligne);
        } while (ligne.size() == 0);

        std::stringstream oss(ligne);

        float x,y,z;
        oss >> x;
        oss >> y;
        oss >> z;
        // TODO : if required read other vertices attributes here
        VEC3 pos(x,y,z);

        unsigned int id = container.insertLine();
        position[id] = pos;

        verticesID.push_back(id);
    }

    // reading tetrahedra
    m_nbFaces.reserve(m_nbVolumes*4);
    m_emb.reserve(m_nbVolumes*12);

    for (unsigned int i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline (fp, ligne);
        } while (ligne.size()==0);

        std::stringstream oss(ligne);
        int n;
        oss >> n; // nb de faces d'un volume ?

        m_nbFaces.push_back(4);

        int s0,s1,s2,s3;

        oss >> s0;
        oss >> s1;
        oss >> s2;
        oss >> s3;

        typename PFP::VEC3 P = position[verticesID[s0]];
        typename PFP::VEC3 A = position[verticesID[s1]];
        typename PFP::VEC3 B = position[verticesID[s2]];
        typename PFP::VEC3 C = position[verticesID[s3]];

        if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::UNDER)
        {
            int ui = s1;
            s1 = s2;
            s2 = ui;
        }

        m_emb.push_back(verticesID[s0]);
        m_emb.push_back(verticesID[s1]);
        m_emb.push_back(verticesID[s2]);
        m_emb.push_back(verticesID[s3]);
    }

    fp.close();
    return true;
}

template <typename PFP>
bool MeshTablesVolume<PFP>::importOFFWithELERegions(const std::string& filenameOFF, const std::string& filenameELE, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3> position =  m_map.template getAttribute<VEC3, VERTEX>("position") ;

    if (!position.isValid())
        position = m_map.template addAttribute<VEC3, VERTEX>("position") ;

    attrNames.push_back(position.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open files
    std::ifstream foff(filenameOFF.c_str(), std::ios::in);
    if (!foff.good())
    {
        CGoGNerr << "Unable to open OFF file " << CGoGNendl;
        return false;
    }

    std::ifstream fele(filenameELE.c_str(), std::ios::in);
    if (!fele.good())
    {
        CGoGNerr << "Unable to open ELE file " << CGoGNendl;
        return false;
    }

    std::string line;

    //OFF reading
    std::getline(foff, line);
    if(line.rfind("OFF") == std::string::npos)
    {
        CGoGNerr << "Problem reading off file: not an off file"<<CGoGNendl;
        CGoGNerr << line << CGoGNendl;
        return false;
    }

    //Reading number of vertex/faces/edges in OFF file
    unsigned int nbe;
    {
        do
        {
            std::getline(foff,line);
        }while(line.size() == 0);

        std::stringstream oss(line);
        oss >> m_nbVertices;
        oss >> nbe;
        oss >> nbe;
        oss >> nbe;
    }

    //Reading number of tetrahedra in ELE file
    unsigned int nbv;
    {
        do
        {
            std::getline(fele,line);
        }while(line.size() == 0);

        std::stringstream oss(line);
        oss >> m_nbVolumes;
        oss >> nbv ;
        oss >> nbv;
    }

    //Reading vertices
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices);

    for(unsigned int i = 0 ; i < m_nbVertices ; ++i)
    {
        do
        {
            std::getline(foff,line);
        }while(line.size() == 0);

        std::stringstream oss(line);

        float x,y,z;
        oss >> x;
        oss >> y;
        oss >> z;
        //we can read colors informations if exists
        VEC3 pos(x,y,z);

        unsigned int id = container.insertLine();
        position[id] = pos;
        verticesID.push_back(id);
    }

    // reading tetrahedra
    m_nbFaces.reserve(m_nbVolumes*4);
    m_emb.reserve(m_nbVolumes*12);

    for(unsigned i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fele,line);
        } while(line.size() == 0);

        std::stringstream oss(line);
        oss >> nbe;

        m_nbFaces.push_back(4);

        int s0,s1,s2,s3;

        oss >> s0;
        oss >> s1;
        oss >> s2;
        oss >> s3;

        typename PFP::VEC3 P = position[verticesID[s0]];
        typename PFP::VEC3 A = position[verticesID[s1]];
        typename PFP::VEC3 B = position[verticesID[s2]];
        typename PFP::VEC3 C = position[verticesID[s3]];

        if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::UNDER)
        {
            int ui= s0;
            s0 = s3;
            s3 = s2;
            s2 = s1;
            s1 = ui;
        }

        m_emb.push_back(verticesID[s0]);
        m_emb.push_back(verticesID[s1]);
        m_emb.push_back(verticesID[s2]);
        m_emb.push_back(verticesID[s3]);
    }

    foff.close();
    fele.close();

    return true;
}

template <typename PFP>
bool MeshTablesVolume<PFP>::importNodeWithELERegions(const std::string& filenameNode, const std::string& filenameELE, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3> position =  m_map.template getAttribute<VEC3, VERTEX>("position") ;

    if (!position.isValid())
        position = m_map.template addAttribute<VEC3, VERTEX>("position") ;

    attrNames.push_back(position.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    //open file
    std::ifstream fnode(filenameNode.c_str(), std::ios::in);
    if (!fnode.good())
    {
        CGoGNerr << "Unable to open file " << filenameNode << CGoGNendl;
        return false;
    }

    std::ifstream fele(filenameELE.c_str(), std::ios::in);
    if (!fele.good())
    {
        CGoGNerr << "Unable to open file " << filenameELE << CGoGNendl;
        return false;
    }

    std::string line;

    //Reading NODE file
    //First line: [# of points] [dimension (must be 3)] [# of attributes] [# of boundary markers (0 or 1)]
    unsigned int nbe;
    {
        do
        {
            std::getline(fnode,line);
        }while(line.size() == 0);

        std::stringstream oss(line);
        oss >> m_nbVertices;
        oss >> nbe;
        oss >> nbe;
        oss >> nbe;
    }

    //Reading number of tetrahedra in ELE file
    unsigned int nbv;
    {
        do
        {
            std::getline(fele,line);
        }while(line.size() == 0);

        std::stringstream oss(line);
        oss >> m_nbVolumes;
        oss >> nbv ;
        oss >> nbv;
    }

    //Reading vertices
    std::map<unsigned int,unsigned int> verticesMapID;

    for(unsigned int i = 0 ; i < m_nbVertices ; ++i)
    {
        do
        {
            std::getline(fnode,line);
        }while(line.size() == 0);

        std::stringstream oss(line);

        int idv;
        oss >> idv;

        float x,y,z;
        oss >> x;
        oss >> y;
        oss >> z;
        //we can read colors informations if exists
        VEC3 pos(x,y,z);

        unsigned int id = container.insertLine();
        position[id] = pos;
        verticesMapID.insert(std::pair<unsigned int, unsigned int>(idv,id));
    }

    // reading tetrahedra
    m_nbFaces.reserve(m_nbVolumes*4);
    m_emb.reserve(m_nbVolumes*12);

    for(unsigned i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fele,line);
        } while(line.size() == 0);

        std::stringstream oss(line);
        oss >> nbe;

        m_nbFaces.push_back(4);

        int s0,s1,s2,s3;

        oss >> s0;
        oss >> s1;
        oss >> s2;
        oss >> s3;

        typename PFP::VEC3 P = position[verticesMapID[s0]];
        typename PFP::VEC3 A = position[verticesMapID[s1]];
        typename PFP::VEC3 B = position[verticesMapID[s2]];
        typename PFP::VEC3 C = position[verticesMapID[s3]];

        if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::UNDER)
        {
            int ui= s0;
            s0 = s3;
            s3 = s2;
            s2 = s1;
            s1 = ui;
        }

        m_emb.push_back(verticesMapID[s0]);
        m_emb.push_back(verticesMapID[s1]);
        m_emb.push_back(verticesMapID[s2]);
        m_emb.push_back(verticesMapID[s3]);
    }

    fnode.close();
    fele.close();

    return true;
}

template <typename PFP>
bool MeshTablesVolume<PFP>::importTetmesh(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3> position =  m_map.template getAttribute<VEC3, VERTEX>("position") ;

    if (!position.isValid())
        position = m_map.template addAttribute<VEC3, VERTEX>("position") ;

    attrNames.push_back(position.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    //open file
    std::ifstream fp(filename.c_str(), std::ios::in);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    std::string line;

    fp >> line;

    if (line!="Vertices")
        CGoGNerr << "Warning tetmesh file problem"<< CGoGNendl;

    fp >> m_nbVertices;

    std::cout << "READ: "<< m_nbVertices << std::endl;

    std::getline (fp, line);

    //reading vertices
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices+1);
    verticesID.push_back(0xffffffff);

    for(unsigned int i = 0; i < m_nbVertices; ++i)
    {
        do
        {
            std::getline (fp, line);
        } while (line.size() == 0);

        std::stringstream oss(line);

        float x,y,z;
        oss >> x;
        oss >> y;
        oss >> z;
        // TODO : if required read other vertices attributes here
        VEC3 pos(x,y,z);

        unsigned int id = container.insertLine();
        position[id] = pos;

        verticesID.push_back(id);
    }

    fp >> line;
    if (line!="Tetrahedra")
        CGoGNerr << "Warning tetmesh file problem"<< CGoGNendl;

    fp >> m_nbVolumes;
    std::getline (fp, line);

    // reading tetrahedra
    m_nbFaces.reserve(m_nbVolumes*4);
    m_emb.reserve(m_nbVolumes*12);

    for(unsigned i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fp,line);
        } while(line.size() == 0);

        std::stringstream oss(line);

        m_nbFaces.push_back(4);

        int s0,s1,s2,s3;

        oss >> s0;
        oss >> s1;
        oss >> s2;
        oss >> s3;

        typename PFP::VEC3 P = position[verticesID[s0]];
        typename PFP::VEC3 A = position[verticesID[s1]];
        typename PFP::VEC3 B = position[verticesID[s2]];
        typename PFP::VEC3 C = position[verticesID[s3]];

        if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::UNDER)
        {
            int ui=s1;
            s1 = s2;
            s2 = ui;
        }

        m_emb.push_back(verticesID[s0]);
        m_emb.push_back(verticesID[s1]);
        m_emb.push_back(verticesID[s2]);
        m_emb.push_back(verticesID[s3]);
    }

    fp.close();
    return true;
}

template <typename PFP>
bool MeshTablesVolume<PFP>::importTs(const std::string& filename, std::vector<std::string>& attrNames)
{
    //
    VertexAttribute<VEC3> position =  m_map.template getAttribute<VEC3, VERTEX>("position") ;

    if (!position.isValid())
        position = m_map.template addAttribute<VEC3, VERTEX>("position") ;

    attrNames.push_back(position.name()) ;

    //
    VertexAttribute<REAL> scalar = m_map.template getAttribute<REAL, VERTEX>("scalar");

    if (!scalar.isValid())
        scalar = m_map.template addAttribute<REAL, VERTEX>("scalar") ;

    attrNames.push_back(scalar.name()) ;

    //
    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::in);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    std::string ligne;

    // reading number of vertices/tetrahedra
    std::getline (fp, ligne);
    std::stringstream oss(ligne);
    oss >> m_nbVertices;
    oss >> m_nbVolumes;

    //reading vertices
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices);

    for(unsigned int i = 0; i < m_nbVertices; ++i)
    {
        do
        {
            std::getline (fp, ligne);
        } while (ligne.size() == 0);

        std::stringstream oss(ligne);

        float x,y,z;
        oss >> x;
        oss >> y;
        oss >> z;

        VEC3 pos(x,y,z);

        unsigned int id = container.insertLine();

        position[id] = pos;
        verticesID.push_back(id);

        float scal;
        oss >> scal;
        scalar[id] = scal;
    }

    //Read and embed all tetrahedrons
    for(unsigned int i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fp,ligne);
        } while(ligne.size() == 0);

        std::stringstream oss(ligne);

        m_nbFaces.push_back(4);

        int s0,s1,s2,s3,nbe;

        oss >> s0;
        oss >> s1;
        oss >> s2;
        oss >> s3;

        typename PFP::VEC3 P = position[verticesID[s0]];
        typename PFP::VEC3 A = position[verticesID[s1]];
        typename PFP::VEC3 B = position[verticesID[s2]];
        typename PFP::VEC3 C = position[verticesID[s3]];

        if(Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::UNDER)
        {
            int ui = s1;
            s1 = s2;
            s2 = ui;
        }


        //if regions are defined use this number
        oss >> nbe; //ignored here

        m_emb.push_back(verticesID[s0]);
        m_emb.push_back(verticesID[s1]);
        m_emb.push_back(verticesID[s2]);
        m_emb.push_back(verticesID[s3]);
    }

    fp.close();
    return true;
}

template <typename PFP>
bool MeshTablesVolume<PFP>::importMSH(const std::string& filename, std::vector<std::string>& attrNames)
{
    //
    VertexAttribute<VEC3> position =  m_map.template getAttribute<VEC3, VERTEX>("position") ;

    if (!position.isValid())
        position = m_map.template addAttribute<VEC3, VERTEX>("position") ;

    attrNames.push_back(position.name()) ;

    //
    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::in);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    std::string ligne;
    unsigned int nbv=0;
    //read $NODE
    std::getline (fp, ligne);

    // reading number of vertices
    std::getline (fp, ligne);
    std::stringstream oss(ligne);
    oss >> m_nbVertices;

    std::map<unsigned int, unsigned int> verticesMapID;

    for(unsigned int i = 0; i < m_nbVertices; ++i)
    {
        do
        {
            std::getline (fp, ligne);
        } while (ligne.size() == 0);

        std::stringstream oss(ligne);

        unsigned int pipo;
        float x,y,z;
        oss >> pipo;
        oss >> x;
        oss >> y;
        oss >> z;

        VEC3 pos(x,y,z);

        unsigned int id = container.insertLine();

        position[id] = pos;
        verticesMapID.insert(std::pair<unsigned int, unsigned int>(pipo,id));
    }

    // ENNODE
    std::getline (fp, ligne);

    // ELM
    std::getline (fp, ligne);

    // reading number of elements
    std::getline (fp, ligne);
    std::stringstream oss2(ligne);
    oss2 >> m_nbVolumes;

    //Read and embed all tetrahedrons
    for(unsigned int i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fp,ligne);
        } while(ligne.size() == 0);

        std::stringstream oss(ligne);

        unsigned int pipo,type_elm,nb;
        oss >> pipo;
        oss >> type_elm;
        oss >> pipo;
        oss >> pipo;
        oss >> nb;

        if ((type_elm==4) && (nb==4))
        {
            m_nbFaces.push_back(4);

            int s0,s1,s2,s3;

            oss >> s0;
            oss >> s1;
            oss >> s2;
            oss >> s3;

            typename PFP::VEC3 P = position[verticesMapID[s0]];
            typename PFP::VEC3 A = position[verticesMapID[s1]];
            typename PFP::VEC3 B = position[verticesMapID[s2]];
            typename PFP::VEC3 C = position[verticesMapID[s3]];

            if(Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::UNDER)
            {
                unsigned int ui= s0;
                s0 = s3;
                s3 = s2;
                s2 = s1;
                s1 = ui;
            }

            unsigned int nbe;
            //if regions are defined use this number
            oss >> nbe; //ignored here

            m_emb.push_back(verticesMapID[s0]);
            m_emb.push_back(verticesMapID[s1]);
            m_emb.push_back(verticesMapID[s2]);
            m_emb.push_back(verticesMapID[s3]);
        }
        else if((type_elm==5) && (nb==8))
        {
            m_nbFaces.push_back(8);
        }
        else
        {
            for (unsigned int j=0; j<nb; ++j)
            {
                unsigned int v;
                fp >> v;
            }
        }

    }
}


} // namespace Import

} // namespace Volume

} // namespace Algo

} // namespace CGoGN
