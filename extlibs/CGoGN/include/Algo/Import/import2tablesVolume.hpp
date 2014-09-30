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
	VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

	if (!position.isValid())
		position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

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
    m_nbVerticesPerVolume.reserve(m_nbVolumes*4);
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

        m_nbVerticesPerVolume.push_back(4);

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
	VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

	if (!position.isValid())
		position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

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
    m_nbVerticesPerVolume.reserve(m_nbVolumes*4);
    m_emb.reserve(m_nbVolumes*12);

    for(unsigned i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fele,line);
        } while(line.size() == 0);

        std::stringstream oss(line);
        oss >> nbe;

        m_nbVerticesPerVolume.push_back(4);

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
	VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

	if (!position.isValid())
		position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

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
    m_nbVerticesPerVolume.reserve(m_nbVolumes*4);
    m_emb.reserve(m_nbVolumes*12);

    for(unsigned i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fele,line);
        } while(line.size() == 0);

        std::stringstream oss(line);
        oss >> nbe;

        m_nbVerticesPerVolume.push_back(4);

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
	VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

	if (!position.isValid())
		position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

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
    m_nbVerticesPerVolume.reserve(m_nbVolumes*4);
    m_emb.reserve(m_nbVolumes*12);

    for(unsigned i = 0; i < m_nbVolumes ; ++i)
    {
        do
        {
            std::getline(fp,line);
        } while(line.size() == 0);

        std::stringstream oss(line);

        m_nbVerticesPerVolume.push_back(4);

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
bool MeshTablesVolume<PFP>::importTs(const std::string& /*filename*/, std::vector<std::string>& /*attrNames*/)
{
//    // open file
//    std::ifstream fp(filename.c_str(), std::ios::in);
//    if (!fp.good())
//    {
//        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
//        return false;
//    }

//    VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

//    if (!position.isValid())
//        position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

//    attrNames.push_back(position.name()) ;

//    VertexAttribute<REAL, MAP> scalar = m_map.template getAttribute<REAL, VERTEX, MAP>("scalar");

//    if (!scalar.isValid())
//        scalar = m_map.template addAttribute<REAL, VERTEX, MAP>("scalar") ;

//    attrNames.push_back(scalar.name()) ;

//    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

//    std::string ligne;

//    // reading number of vertices/tetrahedra
//    std::getline (fp, ligne);
//    std::stringstream oss(ligne);
//    oss >> m_nbVertices;
//    oss >> m_nbVolumes;

//    //reading vertices
//    std::vector<unsigned int> verticesID;
//    verticesID.reserve(m_nbVertices);

//    for(unsigned int i = 0; i < m_nbVertices; ++i)
//    {
//        do
//        {
//            std::getline (fp, ligne);
//        } while (ligne.size() == 0);

//        std::stringstream oss(ligne);

//        float x,y,z;
//        oss >> x;
//        oss >> y;
//        oss >> z;

//        VEC3 pos(x,y,z);

//        unsigned int id = container.insertLine();

//        position[id] = pos;
//        verticesID.push_back(id);

//        float scal;
//        oss >> scal;
//        scalar[id] = scal;
//    }

//    //Read and embed all tetrahedrons
//    for(unsigned int i = 0; i < m_nbVolumes ; ++i)
//    {
//        do
//        {
//            std::getline(fp,ligne);
//        } while(ligne.size() == 0);

//        std::stringstream oss(ligne);

//        m_nbFaces.push_back(4);

//        int s0,s1,s2,s3,nbe;

//        oss >> s0;
//        oss >> s1;
//        oss >> s2;
//        oss >> s3;

//        typename PFP::VEC3 P = position[verticesID[s0]];
//        typename PFP::VEC3 A = position[verticesID[s1]];
//        typename PFP::VEC3 B = position[verticesID[s2]];
//        typename PFP::VEC3 C = position[verticesID[s3]];

//        if(Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::UNDER)
//        {
//            int ui = s1;
//            s1 = s2;
//            s2 = ui;
//        }


//        //if regions are defined use this number
//        oss >> nbe; //ignored here

//        m_emb.push_back(verticesID[s0]);
//        m_emb.push_back(verticesID[s1]);
//        m_emb.push_back(verticesID[s2]);
//        m_emb.push_back(verticesID[s3]);
//    }

//    fp.close();
   return true;
}

template <typename PFP>
bool MeshTablesVolume<PFP>::importMSH(const std::string& filename, std::vector<std::string>& attrNames)
{
	VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

	if (!position.isValid())
		position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

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
            m_nbVerticesPerVolume.push_back(4);

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
            m_nbVerticesPerVolume.push_back(8);
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

//template <typename PFP>
//bool MeshTablesVolume<PFP>::importVBGZ(const std::string& filename, std::vector<std::string>& attrNames)
//{
//	VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

//	if (!position.isValid())
//		position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

//	attrNames.push_back(position.name()) ;

//	//
//	AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

//	//open file
//	igzstream fs(filename.c_str(), std::ios::in|std::ios::binary);
//	if (!fs.good())
//	{
//		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
//		return false;
//	}


//	unsigned int numbers[3];

//	// read nb of points
//	fs.read(reinterpret_cast<char*>(numbers), 3*sizeof(unsigned int));

//	VEC3* bufposi;
//	bufposi = new VEC3[ numbers[0] ];
//	fs.read(reinterpret_cast<char*>(bufposi), numbers[0]*sizeof(VEC3));

//	std::vector<unsigned int> verticesID;
//	verticesID.reserve(numbers[0]);

//	for(unsigned int i = 0; i < numbers[0];++i)
//	{
//		unsigned int id = container.insertLine();
//		position[id] = bufposi[i];
//		verticesID.push_back(id);
//	}
//	delete bufposi;

//	unsigned int* bufTetra=NULL;
//	if (numbers[1] != 0)
//	{
//		bufTetra = new unsigned int[ 4*numbers[1] ];
//		fs.read(reinterpret_cast<char*>(bufTetra), 4*numbers[1]*sizeof(unsigned int));
//	}

//	unsigned int* bufHexa=NULL;
//	if (numbers[2] != 0)
//	{
//		bufHexa = new unsigned int[ 8*numbers[2] ];
//		fs.read(reinterpret_cast<char*>(bufHexa), 8*numbers[2]*sizeof(unsigned int));
//	}
//	CGoGNout << "nb vertices = " << numbers[0];

//	m_nbVertices = numbers[0];
//	m_nbVolumes = numbers[1] + numbers[2];

//	if (numbers[1] > 0)
//	{
//		//Read and embed all tetrahedrons
//		for(unsigned int i = 0; i < numbers[1] ; ++i)
//		{
//			Geom::Vec4ui pt;

//			m_nbFaces.push_back(4);

//			pt[0] = bufTetra[4*i];
//			pt[1] = bufTetra[4*i+1];
//			pt[2] = bufTetra[4*i+2];
//			pt[3] = bufTetra[4*i+3];

//			m_emb.push_back(verticesID[pt[0]]);
//			m_emb.push_back(verticesID[pt[1]]);
//			m_emb.push_back(verticesID[pt[2]]);
//			m_emb.push_back(verticesID[pt[3]]);

//			//end of tetra
//		}
//		CGoGNout << " / nb tetra = " << numbers[1];
//		delete[] bufTetra;
//	}

//	if (numbers[2] > 0)
//	{
//		//Read and embed all tetrahedrons
//		for(unsigned int i = 0; i < numbers[2] ; ++i)
//		{
//			m_nbFaces.push_back(8);
//			// one hexa
//			unsigned int pt[8];
//			pt[0] = bufHexa[8*i];
//			pt[1] = bufHexa[8*i+1];
//			pt[2] = bufHexa[8*i+2];
//			pt[3] = bufHexa[8*i+3];
//			pt[4] = bufHexa[8*i+4];
//			pt[5] = bufHexa[8*i+5];
//			pt[6] = bufHexa[8*i+6];
//			pt[7] = bufHexa[8*i+7];

//			m_emb.push_back(verticesID[pt[0]]);
//			m_emb.push_back(verticesID[pt[1]]);
//			m_emb.push_back(verticesID[pt[2]]);
//			m_emb.push_back(verticesID[pt[3]]);
//			m_emb.push_back(verticesID[pt[4]]);
//			m_emb.push_back(verticesID[pt[5]]);
//			m_emb.push_back(verticesID[pt[6]]);
//			m_emb.push_back(verticesID[pt[7]]);
//			//end of hexa
//		}
//		CGoGNout << " / nb hexa = " << numbers[2];
//		delete[] bufHexa;
//	}

//	return true;
//}

//template <typename PFP>
//bool MeshTablesVolume<PFP>::importVTU(const std::string& filename, std::vector<std::string>& attrNames)
//{
//	VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

//	if (!position.isValid())
//		position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

//	attrNames.push_back(position.name()) ;

//	//
//	AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

//	xmlDocPtr doc = xmlReadFile(filename.c_str(), NULL, 0);
//	xmlNodePtr vtu_node = xmlDocGetRootElement(doc);

//	//	std::cout << " NAME "<<vtu_node->name << std::endl;

//	xmlChar *prop = xmlGetProp(vtu_node, BAD_CAST "type");
//	//	std::cout << "type = "<< prop << std::endl;

//	xmlNode* grid_node = vtu_node->children;
//	while (strcmp((char*)(grid_node->name),(char*)"UnstructuredGrid")!=0)
//		grid_node = grid_node->next;

//	xmlNode* piece_node = grid_node->children;
//	while (strcmp((char*)(piece_node->name),(char*)"Piece")!=0)
//		piece_node = piece_node->next;

//	prop = xmlGetProp(piece_node, BAD_CAST "NumberOfPoints");
//	m_nbVertices = atoi((char*)(prop));

//	prop = xmlGetProp(piece_node, BAD_CAST "NumberOfCells");
//	m_nbVolumes = atoi((char*)(prop));

//	std::cout << "Number of points = "<< m_nbVertices<< std::endl;
//	std::cout << "Number of cells = "<< m_nbVolumes << std::endl;

//	xmlNode* points_node = piece_node->children;
//	while (strcmp((char*)(points_node->name),(char*)"Points")!=0)
//		points_node = points_node->next;

//	points_node = points_node->children;
//	while (strcmp((char*)(points_node->name),(char*)"DataArray")!=0)
//		points_node = points_node->next;

//	std::vector<unsigned int> verticesID;
//	verticesID.reserve(m_nbVertices);

//	std::stringstream ss((char*)(xmlNodeGetContent(points_node->children)));
//	for (unsigned int i=0; i< m_nbVertices; ++i)
//	{
//		typename PFP::VEC3 P;
//		ss >> P[0]; ss >> P[1]; ss >> P[2];
//		unsigned int id = container.insertLine();
//		position[id] = P;
//		verticesID.push_back(id);
//	}

//	xmlNode* cell_node = piece_node->children;
//	while (strcmp((char*)(cell_node->name),(char*)"Cells")!=0)
//		cell_node = cell_node->next;

//	std::cout <<"CELL NODE = "<< cell_node->name << std::endl;


//	std::vector<unsigned char> typeVols;
//	typeVols.reserve(m_nbVolumes);
//	std::vector<unsigned int> offsets;
//	offsets.reserve(m_nbVolumes);
//	std::vector<unsigned int> indices;
//	indices.reserve(m_nbVolumes*4);

//	for (xmlNode* x_node = cell_node->children; x_node!=NULL; x_node = x_node->next)
//	{
//		while ((x_node!=NULL) && (strcmp((char*)(x_node->name),(char*)"DataArray")!=0))
//			x_node = x_node->next;

//		if (x_node == NULL)
//			break;
//		else
//		{
//			xmlChar* type = xmlGetProp(x_node, BAD_CAST "Name");

//			if (strcmp((char*)(type),(char*)"connectivity")==0)
//			{
//				std::stringstream ss((char*)(xmlNodeGetContent(x_node->children)));
//				while (!ss.eof())
//				{
//					unsigned int ind;
//					ss >> ind;
//					indices.push_back(ind);
//				}
//			}
//			if (strcmp((char*)(type),(char*)"offsets")==0)
//			{
//				std::stringstream ss((char*)(xmlNodeGetContent(x_node->children)));
//				for (unsigned int i=0; i< m_nbVolumes; ++i)
//				{
//					unsigned int o;
//					ss >> o;
//					offsets.push_back(o);
//				}
//			}
//			if (strcmp((char*)(type),(char*)"types")==0)
//			{
//				bool unsupported = false;
//				std::stringstream ss((char*)(xmlNodeGetContent(x_node->children)));
//				for (unsigned int i=0; i< m_nbVolumes; ++i)
//				{
//					unsigned int t;
//					ss >> t;
//					if ((t != 12) && (t!= 10))
//					{
//						unsupported = true;
//						typeVols.push_back(0);
//					}
//					else
//					{
//						typeVols.push_back((unsigned char)t);
//					}
//				}
//				if (unsupported)
//					CGoGNerr << "warning, some unsupported volume cell types"<< CGoGNendl;
//			}
//		}
//	}

//	xmlFreeDoc(doc);

//	unsigned int currentOffset = 0;
//	for (unsigned int i=0; i< m_nbVolumes; ++i)
//	{
//		if (typeVols[i]==12)
//		{
//			m_nbFaces.push_back(8);

//			unsigned int pt[8];
//			pt[0] = indices[currentOffset];
//			pt[1] = indices[currentOffset+1];
//			pt[2] = indices[currentOffset+2];
//			pt[3] = indices[currentOffset+3];
//			pt[4] = indices[currentOffset+4];
//			typename PFP::VEC3 P = position[verticesID[indices[currentOffset+4]]];
//			typename PFP::VEC3 A = position[verticesID[indices[currentOffset  ]]];
//			typename PFP::VEC3 B = position[verticesID[indices[currentOffset+1]]];
//			typename PFP::VEC3 C = position[verticesID[indices[currentOffset+2]]];

//			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
//			{

//				pt[0] = indices[currentOffset+3];
//				pt[1] = indices[currentOffset+2];
//				pt[2] = indices[currentOffset+1];
//				pt[3] = indices[currentOffset+0];
//				pt[4] = indices[currentOffset+7];
//				pt[5] = indices[currentOffset+6];
//				pt[6] = indices[currentOffset+5];
//				pt[7] = indices[currentOffset+4];
//			}
//			else
//			{
//				pt[0] = indices[currentOffset+0];
//				pt[1] = indices[currentOffset+1];
//				pt[2] = indices[currentOffset+2];
//				pt[3] = indices[currentOffset+3];
//				pt[4] = indices[currentOffset+4];
//				pt[5] = indices[currentOffset+5];
//				pt[6] = indices[currentOffset+6];
//				pt[7] = indices[currentOffset+7];
//			}

//			m_emb.push_back(verticesID[pt[0]]);
//			m_emb.push_back(verticesID[pt[1]]);
//			m_emb.push_back(verticesID[pt[2]]);
//			m_emb.push_back(verticesID[pt[3]]);
//			m_emb.push_back(verticesID[pt[4]]);
//			m_emb.push_back(verticesID[pt[5]]);
//			m_emb.push_back(verticesID[pt[6]]);
//			m_emb.push_back(verticesID[pt[7]]);

//		}
//		else if (typeVols[i]==10)
//		{
//			m_nbFaces.push_back(4);

//			Geom::Vec4ui pt;
//			pt[0] = indices[currentOffset];
//			pt[1] = indices[currentOffset+1];
//			pt[2] = indices[currentOffset+2];
//			pt[3] = indices[currentOffset+3];

//			typename PFP::VEC3 P = position[verticesID[pt[0]]];
//			typename PFP::VEC3 A = position[verticesID[pt[1]]];
//			typename PFP::VEC3 B = position[verticesID[pt[2]]];
//			typename PFP::VEC3 C = position[verticesID[pt[3]]];

//			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
//			{
//				unsigned int ui=pt[1];
//				pt[1] = pt[2];
//				pt[2] = ui;
//			}

//			m_emb.push_back(verticesID[pt[0]]);
//			m_emb.push_back(verticesID[pt[1]]);
//			m_emb.push_back(verticesID[pt[2]]);
//			m_emb.push_back(verticesID[pt[3]]);
//		}
//		currentOffset = offsets[i];
//	}

//	return true;
//}

} // namespace Import

} // namespace Volume

} // namespace Algo

} // namespace CGoGN
