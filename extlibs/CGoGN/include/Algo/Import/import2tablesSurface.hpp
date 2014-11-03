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
* FITNESS FOR A PARTICULAR PURVEC3E. See the GNU Lesser General Public License  *
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

#include "Algo/Import/importPlyData.h"
#include "Algo/Geometry/boundingbox.h"
#include "Topology/generic/autoAttributeHandler.h"

#include "Algo/Modelisation/voxellisation.h"

#include "Algo/Import/AHEM.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{

template<typename PFP>
bool MeshTablesSurface<PFP>::importMesh(const std::string& filename, std::vector<std::string>& attrNames)
{
    ImportType kind = getFileType(filename);

    attrNames.clear() ;

    switch (kind)
    {
    case TRIAN:
        CGoGNout << "TYPE: TRIAN" << CGoGNendl;
        return importTrian(filename, attrNames);
        break;
    case TRIANBGZ:
        CGoGNout << "TYPE: TRIANBGZ" << CGoGNendl;
        return importTrianBinGz(filename, attrNames);
        break;
    case OFF:
        CGoGNout << "TYPE: OFF" << CGoGNendl;
        return importOff(filename, attrNames);
        break;
    case MESHBIN:
        CGoGNout << "TYPE: MESHBIN" << CGoGNendl;
        return importMeshBin(filename, attrNames);
        break;
    case PLY:
        CGoGNout << "TYPE: PLY" << CGoGNendl;
        return importPly(filename, attrNames);
        break;
        /*	case PLYPTM:
        CGoGNout << "TYPE: PLYPTM" << CGoGNendl;
        return importPlyPTM(filename, attrNames);
        break;
*/	case PLYSLFgeneric:
        CGoGNout << "TYPE: PLYSLFgeneric" << CGoGNendl;
        return importPlySLFgeneric(filename, attrNames);
        break;
    case PLYSLFgenericBin:
        CGoGNout << "TYPE: PLYSLFgenericBin" << CGoGNendl;
        return importPlySLFgenericBin(filename, attrNames);
        break;
    case OBJ:
        CGoGNout << "TYPE: OBJ" << CGoGNendl;
        return importObj(filename, attrNames);
        break;
    case AHEM:
        CGoGNout << "TYPE: AHEM" << CGoGNendl;
        return importAHEM(filename, attrNames);
        break;
    case STLB:
        CGoGNout << "TYPE: STLB" << CGoGNendl;
        return importSTLBin(filename, attrNames);
        break;
    case STL:
        CGoGNout << "TYPE: STL" << CGoGNendl;
        return importSTLAscii(filename, attrNames);
        break;
    default:
#ifdef WITH_ASSIMP
        CGoGNout << "TYPE: ASSIMP" << CGoGNendl;
        return importASSIMP(filename, attrNames);
#else
        CGoGNout << "unsupported file type"<< CGoGNendl;
#endif
        break;
    }
    return false;
}

template<typename PFP>
bool MeshTablesSurface<PFP>::importTrian(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::in);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    // read nb of points
    fp >> m_nbVertices;

    // read points
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices);

    for (unsigned int i = 0; i < m_nbVertices; ++i)
    {
        VEC3 pos;
        fp >> pos[0];
        fp >> pos[1];
        fp >> pos[2];
        unsigned int id = container.insertLine();
        positions[id] = pos;
        verticesID.push_back(id);
    }

    // read nb of faces
    fp >> m_nbFaces;
    m_nbEdges.reserve(m_nbFaces);
    m_emb.reserve(3*m_nbFaces);

    // read indices of faces
    for (unsigned int i = 0; i < m_nbFaces; ++i)
    {
        m_nbEdges.push_back(3);
        // read the three vertices of triangle
        int pt;
        fp >> pt;
        m_emb.push_back(verticesID[pt]);
        fp >> pt;
        m_emb.push_back(verticesID[pt]);
        fp >> pt;
        m_emb.push_back(verticesID[pt]);

        // neighbour not always good in files !!
        int neigh;
        fp >> neigh;
        fp >> neigh;
        fp >> neigh;
    }

    fp.close();
    return true;
}

template<typename PFP>
bool MeshTablesSurface<PFP>::importTrianBinGz(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    igzstream fs(filename.c_str(), std::ios::in|std::ios::binary);

    if (!fs.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    // read nb of points
    fs.read(reinterpret_cast<char*>(&m_nbVertices), sizeof(int));

    // read points
    std::vector<unsigned int> verticesID;
    {	// juste pour limiter la portee des variables
        verticesID.reserve(m_nbVertices);
        float* buffer = new float[m_nbVertices*3];
        fs.read(reinterpret_cast<char*>(buffer), 3*m_nbVertices*sizeof(float));
        float *ptr = buffer;
        for (unsigned int i = 0; i < m_nbVertices; ++i)
        {
            VEC3 pos;
            pos[0]= *ptr++;
            pos[1]= *ptr++;
            pos[2]= *ptr++;

            unsigned int id = container.insertLine();
            positions[id] = pos;

            verticesID.push_back(id);
        }
        delete[] buffer;
    }

    // read nb of faces
    fs.read(reinterpret_cast<char*>(&m_nbFaces), sizeof(int));
    m_nbEdges.reserve(m_nbFaces);
    m_emb.reserve(3*m_nbFaces);

    // read indices of faces
    {	// juste pour limiter la portee des variables
        int* buffer = new int[m_nbFaces*6];
        fs.read(reinterpret_cast<char*>(buffer),6*m_nbFaces*sizeof(float));
        int *ptr = buffer;

        for (unsigned int i = 0; i < m_nbFaces; i++)
        {
            m_nbEdges.push_back(3);
            m_emb.push_back(verticesID[*ptr++]);
            m_emb.push_back(verticesID[*ptr++]);
            m_emb.push_back(verticesID[*ptr++]);
        }
    }

    fs.close();
    return true;
}

template<typename PFP>
bool MeshTablesSurface<PFP>::importOff(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions = m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::in);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    std::string ligne;

    // lecture de OFF
    std::getline (fp, ligne);
    if (ligne.rfind("OFF") == std::string::npos)
    {
        CGoGNerr << "Problem reading off file: not an off file" << CGoGNendl;
        CGoGNerr << ligne << CGoGNendl;
        return false;
    }

    // lecture des nombres de sommets/faces/aretes
    int nbe;
    {
        do
        {
            std::getline (fp, ligne);
        } while (ligne.size() == 0);

        std::stringstream oss(ligne);
        oss >> m_nbVertices;
        oss >> m_nbFaces;
        oss >> nbe;
    }

    //lecture sommets
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices);
    for (unsigned int i = 0; i < m_nbVertices;++i)
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
        // on peut ajouter ici la lecture de couleur si elle existe
        VEC3 pos(x,y,z);

        unsigned int id = container.insertLine();
        positions[id] = pos;

        verticesID.push_back(id);
    }

    // lecture faces
    // normalement nbVertices*8 devrait suffire largement
    m_nbEdges.reserve(m_nbFaces);
    m_emb.reserve(m_nbVertices*8);

    for (unsigned int i = 0; i < m_nbFaces; ++i)
    {
        do
        {
            std::getline (fp, ligne);
        } while (ligne.size() == 0);

        std::stringstream oss(ligne);
        unsigned int n;
        oss >> n;
        m_nbEdges.push_back(n);
        for (unsigned int j = 0; j < n; ++j)
        {
            int index; // index du plongement
            oss >> index;
            m_emb.push_back(verticesID[index]);
        }
        // on peut ajouter ici la lecture de couleur si elle existe
    }

    fp.close();
    return true;
}

template<typename PFP>
bool MeshTablesSurface<PFP>::importVoxellisation(Algo::Surface::Modelisation::Voxellisation& voxellisation, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions = m_map.template getAttribute<VEC3, VERTEX>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // lecture des nombres de sommets/faces
    m_nbVertices = voxellisation.getNbSommets();
    m_nbFaces = voxellisation.getNbFaces();

    //lecture sommets
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices);
    for (unsigned int i = 0; i < m_nbVertices;++i)
    {
        unsigned int id = container.insertLine();
        positions[id] = voxellisation.m_sommets[i];

        verticesID.push_back(id);
    }

    // lecture faces
    // normalement nbVertices*8 devrait suffire largement
    m_nbEdges.reserve(m_nbFaces);
    m_emb.reserve(m_nbVertices*8);

    for (unsigned int i = 0; i < m_nbFaces*4-3; i=i+4)
    {
        m_nbEdges.push_back(4); //Toutes les faces ont 4 côtés (de par leur construction)
        for (unsigned int j = 0; j < 4; ++j)
        {
            m_emb.push_back(verticesID[voxellisation.m_faces[i+j]]);
        }
    }

    return true;
}

template<typename PFP>
bool MeshTablesSurface<PFP>::importMeshBin(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions = m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
    {
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;
    }

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::in | std::ios::binary);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    // Read header
    unsigned int Fmin, Fmax ;
    fp.read((char*)&m_nbVertices, sizeof(unsigned int)) ;
    fp.read((char*)&m_nbFaces, sizeof(unsigned int)) ;
    fp.read((char*)&Fmin, sizeof(unsigned int)) ;
    fp.read((char*)&Fmax, sizeof(unsigned int)) ;

    assert((Fmin == 3 && Fmax == 3) || !"Only triangular faces are handled yet") ;

    // Read foreach vertex
    std::vector<unsigned int> verticesID ;
    verticesID.reserve(m_nbVertices) ;

    for (unsigned int vxNum = 0 ; vxNum < m_nbVertices ; ++vxNum)
    {
        Geom::Vec3f pos ;
        fp.read((char*) &pos[0], sizeof(float)) ;
        fp.read((char*) &pos[1], sizeof(float)) ;
        fp.read((char*) &pos[2], sizeof(float)) ;

        unsigned int id = container.insertLine() ;
        positions[id] = pos ;

        verticesID.push_back(id) ;
    }

    // Read foreach face
    m_nbEdges.reserve(m_nbFaces) ;
    m_emb.reserve(m_nbVertices * 8) ;

    for (unsigned int fNum = 0; fNum < m_nbFaces; ++fNum)
    {
        const unsigned int faceSize = 3 ;
        fp.read((char*) &faceSize, sizeof(float)) ;
        m_nbEdges.push_back(faceSize) ;

        for (unsigned int i = 0 ; i < faceSize; ++i)
        {
            unsigned int index ; // index of embedding
            fp.read((char*) &index, sizeof(unsigned int)) ;
            m_emb.push_back(verticesID[index]) ;
        }
    }

    fp.close() ;
    return true ;
}


template <typename PFP>
bool MeshTablesSurface<PFP>::importObj(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::binary);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    //	fp.seekg(0, std::ios::end);
    //	int ab = fp.tellg();
    //	fp.seekg(0, std::ios::beg);
    //	int ac = fp.tellg();

    std::string ligne;
    std::string tag;

    do
    {
        fp >> tag;
        std::getline (fp, ligne);
    }while (tag != std::string("v"));

    // lecture des sommets
    std::vector<unsigned int> verticesID;
    verticesID.reserve(102400); // on tape large (400Ko wahouuuuu !!)

    unsigned int i = 0;
    do
    {
        if (tag == std::string("v"))
        {
            std::stringstream oss(ligne);

            float x,y,z;
            oss >> x;
            oss >> y;
            oss >> z;

            VEC3 pos(x,y,z);

            unsigned int id = container.insertLine();
            positions[id] = pos;

            verticesID.push_back(id);
            i++;
        }

        fp >> tag;
        std::getline(fp, ligne);
    } while (!fp.eof());

    m_nbVertices = verticesID.size();

    // close/clear/open only way to go back to beginning of file
    fp.close();
    fp.clear();
    fp.open(filename.c_str());

    do
    {
        fp >> tag;
        std::getline (fp, ligne);
    } while (tag != std::string("f"));

    m_nbEdges.reserve(verticesID.size()*2);
    m_emb.reserve(verticesID.size()*8);

    std::vector<int> table;
    table.reserve(64); // NBV cotes pour une face devrait suffire
    m_nbFaces = 0;
    do
    {
        if (tag == std::string("f")) // lecture d'une face
        {
            std::stringstream oss(ligne);
            table.clear();
            while (!oss.eof())  // lecture de tous les indices
            {
                std::string str;
                oss >> str;

                unsigned int ind = 0;

                while ((ind<str.length()) && (str[ind]!='/'))
                    ind++;

                if (ind > 0)
                {
                    long index;
                    std::stringstream iss(str.substr(0, ind));
                    iss >> index;
                    table.push_back(index);
                }
            }

            unsigned int n = table.size();
            m_nbEdges.push_back(short(n));
            for (unsigned int j = 0; j < n; ++j)
            {
                int index = table[j] - 1; // les index commencent a 1 (boufonnerie d'obj ;)
                m_emb.push_back(verticesID[index]);
            }
            m_nbFaces++;
        }
        fp >> tag;
        std::getline(fp, ligne);
    } while (!fp.eof());

    fp.close ();
    return true;
}

template<typename PFP>
bool MeshTablesSurface<PFP>::importPly(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    PlyImportData pid;

    if (! pid.read_file(filename) )
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    VertexAttribute<VEC3, MAP> colors = m_map.template getAttribute<VEC3, VERTEX, MAP>("color") ;
    if (pid.hasColors())
    {
        if(!colors.isValid())
            colors = m_map.template addAttribute<VEC3, VERTEX, MAP>("color") ;
        attrNames.push_back(colors.name()) ;
    }

    // lecture des nombres de sommets/aretes/faces
    m_nbVertices = pid.nbVertices();
    m_nbFaces = pid.nbFaces();

    //lecture sommets
    std::vector<unsigned int> verticesID;
    verticesID.reserve(m_nbVertices);
    for (unsigned int i = 0; i < m_nbVertices; ++i)
    {
        VEC3 pos;
        pid.vertexPosition(i, pos);

        unsigned int id = container.insertLine();
        positions[id] = pos;

        if (pid.hasColorsUint8())
        {
            Geom::Vector<3, unsigned char> col ;
            pid.vertexColorUint8(i, col) ;
            colors[id][0] = col[0] ;
            colors[id][1] = col[1] ;
            colors[id][2] = col[2] ;
            colors[id] /= 255.0 ;
        }

        if (pid.hasColorsFloat32())
        {
            Geom::Vector<3, float> col ;
            pid.vertexColorFloat32(i, col) ;
            colors[id][0] = col[0] ;
            colors[id][1] = col[1] ;
            colors[id][2] = col[2] ;
        }


        verticesID.push_back(id);
    }

    m_nbEdges.reserve(m_nbFaces);
    m_emb.reserve(m_nbVertices*8);

    for (unsigned int i = 0 ; i < m_nbFaces ; ++i)
    {
        unsigned int n = pid.getFaceValence(i);
        m_nbEdges.push_back(n);
        int* indices = pid.getFaceIndices(i);
        for (unsigned int j = 0; j < n; ++j)
        {
            m_emb.push_back(verticesID[indices[j]]);
        }
    }

    return true;
}

/**
 * Import plySLF (K Vanhoey generic format).
 * It can handle bivariable polynomials and spherical harmonics of any degree and returns the appropriate attrNames
 * @param filename the file to import;
 * @param attrNames reference that will be filled with the attribute names
 * the number of attrNames returned depends on the degree of the polynomials / level of the SH :
 *  - 1 attrName for geometric position (VEC3) : name = "position" ;
 *  - 3 attrNames for local frame (3xVEC3) : names are "frameT" (Tangent), "frameB" (Binormal) and "frameN" (Normal) ;
 *  - N attrNames for the function coefficients (NxVEC3) : N RGB coefficients being successively the constants, the linears (v then u), the quadratics, etc. :  : a0 + a1*v + a2*u + a3*u*v + a4*v^2 + a5*u^2.
 *  Their names are : "SLFcoefs<i>" (where <i> is a number from 0 to N-1).
 * N = 1 for constant polynomial,
 * N = 3 for linear polynomial,
 * N = 6 for quadratic polynomial,
 * N = 10 for cubic degree polynomial,
 * N = 15 for 4th degree polynomial,
 *
 * N = l*l for SH of level l,
 * ...
 *  - K remaining attrNames named "remainderNo<k>" where k is an integer from 0 to K-1.
 * @return bool : success.
 */
template <typename PFP>
bool MeshTablesSurface<PFP>::importPlySLFgeneric(const std::string& filename, std::vector<std::string>& attrNames)
{
    // Open file
    std::ifstream fp(filename.c_str(), std::ios::in) ;
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
        return false ;
    }

    // Read quantities : #vertices, #faces, #properties, degree of polynomials
    std::string tag ;

    fp >> tag;
    if (tag != std::string("ply")) // verify file type
    {
        CGoGNerr << filename << " is not a ply file !" <<  CGoGNout ;
        return false ;
    }

    do // go to #vertices
    {
        fp >> tag ;
    } while (tag != std::string("vertex")) ;
    unsigned int nbVertices ;
    fp >> nbVertices ; // Read #vertices

    bool position = false ;
    bool tangent = false ;
    bool binormal = false ;
    bool normal = false ;
    bool PTM = false ;
    bool SH = false ;
    unsigned int nbProps = 0 ;		// # properties
    unsigned int nbCoefs = 0 ; 	// # coefficients
    do // go to #faces and count #properties
    {
        fp >> tag ;
        if (tag == std::string("property"))
            ++nbProps ;
        else if (tag == std::string("x") || tag == std::string("y") || tag == std::string("z"))
            position = true ;
        //else if (tag == std::string("tx") || tag == std::string("ty") || tag == std::string("tz"))
        else if (tag == std::string("frameT_0") || tag == std::string("frameT_1") || tag == std::string("frameT_2"))
            tangent = true ;
        //else if (tag == std::string("bx") || tag == std::string("by") || tag == std::string("bz"))
        else if (tag == std::string("frameB_0") || tag == std::string("frameB_1") || tag == std::string("frameB_2"))
            binormal = true ;
        //else if (tag == std::string("nx") || tag == std::string("ny") || tag == std::string("nz"))
        else if (tag == std::string("frameN_0") || tag == std::string("frameN_1") || tag == std::string("frameN_2"))
            normal = true ;
        // else if (tag.substr(0,1) == std::string("C") && tag.substr(2,1) == std::string("_"))
        else if ((tag.length() > 8) && tag.find("_") != std::string::npos)
        {
            if (tag.substr(0,7) == std::string("PBcoefs"))
            {
                PTM = true ;
                ++nbCoefs ;
            }
            else if (tag.substr(0,7) == std::string("SHcoefs"))
            {
                SH = true ;
                ++nbCoefs ;
            }
        }
    } while (tag != std::string("face")) ;
    unsigned int nbRemainders = nbProps ;		// # remaining properties
    nbRemainders -= nbCoefs + 3*(position==true) + 3*(tangent==true) + 3*(binormal==true) + 3*(normal==true) ;
    nbCoefs /= 3 ;

    fp >> m_nbFaces ; // Read #vertices

    do // go to end of header
    {
        fp >> tag ;
    } while (tag != std::string("end_header")) ;

    // Define containers
    VertexAttribute<VEC3, MAP> positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;
    ;
    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;
    attrNames.push_back(positions.name()) ;

    VertexAttribute<VEC3, MAP> *frame = new VertexAttribute<VEC3, MAP>[3] ;
    frame[0] = m_map.template addAttribute<VEC3, VERTEX, MAP>("frameT") ; // Tangent
    frame[0] = m_map.template addAttribute<VEC3, VERTEX, MAP>("frameB") ; // Binormal
    frame[0] = m_map.template addAttribute<VEC3, VERTEX, MAP>("frameN") ; // Normal
    attrNames.push_back(frame[0].name()) ;
    attrNames.push_back(frame[1].name()) ;
    attrNames.push_back(frame[2].name()) ;

    VertexAttribute<VEC3, MAP> *PBcoefs = NULL, *SHcoefs = NULL ;
    if (PTM)
    {
        PBcoefs = new VertexAttribute<VEC3, MAP>[nbCoefs] ;
        for (unsigned int i = 0 ; i < nbCoefs ; ++i)
        {
            std::stringstream name ;
            name << "PBcoefs" << i ;
            PBcoefs[i] = m_map.template addAttribute<VEC3, VERTEX, MAP>(name.str()) ;
            attrNames.push_back(PBcoefs[i].name()) ;
        }
    }

    if (SH)
    {
        SHcoefs = new VertexAttribute<VEC3, MAP>[nbCoefs] ;
        for (unsigned int i = 0 ; i < nbCoefs ; ++i)
        {
            std::stringstream name ;
            name << "SHcoefs" << i ;
            SHcoefs[i] = m_map.template addAttribute<VEC3, VERTEX, MAP>(name.str()) ;
            attrNames.push_back(SHcoefs[i].name()) ;
        }
    }

    VertexAttribute<REAL, MAP> *remainders = new VertexAttribute<REAL, MAP>[nbRemainders] ;
    for (unsigned int i = 0 ; i < nbRemainders ; ++i)
    {
        std::stringstream name ;
        name << "remainderNo" << i ;
        remainders[i] = m_map.template addAttribute<REAL, VERTEX, MAP>(name.str()) ;
        attrNames.push_back(remainders[i].name()) ;
    }

    // Read vertices
    std::vector<unsigned int> verticesID ;
    verticesID.reserve(nbVertices) ;

    float* properties = new float[nbProps] ;
    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;
    for (unsigned int i = 0 ; i < nbVertices ; ++i) // Read and store properties for current vertex
    {
        unsigned int id = container.insertLine() ;
        verticesID.push_back(id) ;

        for (unsigned int j = 0 ; j < nbProps ; ++j) // get all properties
            fp >> properties[j] ;

        positions[id] = VEC3(properties[0],properties[1],properties[2]) ; // position
        for (unsigned int k = 0 ; k < 3 ; ++k) // frame
            for (unsigned int l = 0 ; l < 3 ; ++l)
                frame[k][id][l] = properties[3+(3*k+l)] ;
        for (unsigned int l = 0 ; l < nbCoefs ; ++l) // coefficients
            for (unsigned int k = 0 ; k < 3 ; ++k)
            {
                if (PTM)
                    PBcoefs[l][id][k] = (typename PFP::REAL)(properties[12+(3*l+k)]) ;
                else /* if SH */
                    SHcoefs[l][id][k] = (typename PFP::REAL)(properties[12+(3*l+k)]) ;
            }
        unsigned int cur = 12+3*nbCoefs ;
        for (unsigned int k = 0 ; k < nbRemainders ; ++k) // remaining data
            remainders[k][id] = properties[cur + k] ;
    }
    m_nbVertices = verticesID.size() ;
    delete[] properties ;

    // Read faces index
    m_nbEdges.reserve(m_nbFaces) ;
    m_emb.reserve(3 * m_nbFaces) ;
    for (unsigned int i = 0 ; i < m_nbFaces ; ++i)
    {
        // read the indices of vertices for current face
        unsigned int nbEdgesForFace ;
        fp >> nbEdgesForFace ;
        m_nbEdges.push_back(nbEdgesForFace);

        unsigned int vertexID ;
        for (unsigned int j=0 ; j < nbEdgesForFace ; ++j)
        {
            fp >> vertexID ;
            m_emb.push_back(verticesID[vertexID]);
        }
    }

    // Close file
    fp.close() ;

    return true ;
}

template <typename PFP>
bool MeshTablesSurface<PFP>::importPlySLFgenericBin(const std::string& filename, std::vector<std::string>& attrNames)
{
    // Open file
    std::ifstream fp(filename.c_str(), std::ios::in | std::ios::binary) ;
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
        return false ;
    }

    // Read quantities : #vertices, #faces, #properties, degree of polynomials
    std::string tag ;

    fp >> tag;
    if (tag != std::string("ply")) // verify file type
    {
        CGoGNerr << filename << " is not a ply file !" <<  CGoGNout ;
        return false ;
    }

    do // go to #vertices
    {
        fp >> tag ;
    } while (tag != std::string("vertex")) ;
    unsigned int nbVertices ;
    fp >> nbVertices ; // Read #vertices

    bool position = false ;
    bool tangent = false ;
    bool binormal = false ;
    bool normal = false ;
    bool PTM = false ;
    bool SH = false ;
    unsigned int propSize = 0 ;
    unsigned int nbProps = 0 ;		// # properties
    unsigned int nbCoefs = 0 ; 	// # coefficients
    do // go to #faces and count #properties
    {
        fp >> tag ;
        if (tag == std::string("property"))
            ++nbProps ;
        else if (tag == std::string("int8") || tag == std::string("uint8"))
        {
            if (propSize < 2)
            {
                propSize = 1 ;
                std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: only float64 is yet handled" << std::endl ;
                assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: only float64 is yet handled") ;			}
            else
            {
                std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled" << std::endl ;
                assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled") ;
            }
        }
        else if (tag == std::string("int16") || tag == std::string("uint16"))
        {
            if (propSize == 0 || propSize == 2)
            {
                propSize = 2 ;
                std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: only float64 is yet handled" << std::endl ;
                assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: only float64 is yet handled") ;			}
            else
            {
                std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled" << std::endl ;
                assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled") ;
            }
        }
        else if (tag == std::string("int32") || tag == std::string("float32") || tag == std::string("uint32"))
        {
            if (propSize == 0 || propSize == 4)
            {
                propSize = 4 ;
                std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: only float64 is yet handled" << std::endl ;
                assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: only float64 is yet handled") ;
            }
            else
            {
                std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled" << std::endl ;
                assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled") ;
            }
        }
        else if (tag == std::string("int64") || tag == std::string("float64"))
        {
            if (propSize == 0 || propSize == 8)
            {
                propSize = 8 ;
                //std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: only float32 is yet handled" << std::endl ;
                //assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: only float32 is yet handled") ;
            }
            else
            {
                std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled" << std::endl ;
                assert(!"MeshTablesSurface<PFP>::importPlySLFgenericBin: all properties should be same size: otherwise not handled") ;
            }
        }
        else if (tag == std::string("x") || tag == std::string("y") || tag == std::string("z"))
            position = true ;
        //else if (tag == std::string("tx") || tag == std::string("ty") || tag == std::string("tz"))
        else if (tag == std::string("frameT_0") || tag == std::string("frameT_1") || tag == std::string("frameT_2"))
            tangent = true ;
        //else if (tag == std::string("bx") || tag == std::string("by") || tag == std::string("bz"))
        else if (tag == std::string("frameB_0") || tag == std::string("frameB_1") || tag == std::string("frameB_2"))
            binormal = true ;
        //else if (tag == std::string("nx") || tag == std::string("ny") || tag == std::string("nz"))
        else if (tag == std::string("frameN_0") || tag == std::string("frameN_1") || tag == std::string("frameN_2"))
            normal = true ;
        // else if (tag.substr(0,1) == std::string("C") && tag.substr(2,1) == std::string("_"))
        else if ((tag.length() > 8) && tag.find("_") != std::string::npos)
        {
            if (tag.substr(0,7) == std::string("PBcoefs"))
            {
                PTM = true ;
                ++nbCoefs ;
            }
            else if (tag.substr(0,7) == std::string("SHcoefs"))
            {
                SH = true ;
                ++nbCoefs ;
            }
        }
    } while (tag != std::string("face")) ;
    unsigned int nbRemainders = nbProps ;		// # remaining properties
    nbRemainders -= nbCoefs + 3*(position==true) + 3*(tangent==true) + 3*(binormal==true) + 3*(normal==true) ;
    nbCoefs /= 3 ;

    assert(!(SH && PTM) || !"MeshTablesSurface<PFP>::importPlySLFgenericBin: Confusing functional colors since both SLF and RF are defined.") ;
    if (SH && PTM)
        std::cerr << "MeshTablesSurface<PFP>::importPlySLFgenericBin: Confusing functional colors since both SLF and RF are defined." << std::endl ;

    fp >> m_nbFaces ; // Read #vertices

    do // go to end of header
    {
        fp >> tag ;
    } while (tag != std::string("end_header")) ;

    char* endline = new char[1] ;
    fp.read(endline, sizeof(char)) ;
    if (*endline == '\r') // for windows
        fp.read(endline, sizeof(char)) ;
    assert(*endline == '\n') ;
    delete[] endline ;

    // Define containers
    VertexAttribute<VEC3, MAP> positions = m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;
    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;
    attrNames.push_back(positions.name()) ;

    VertexAttribute<VEC3, MAP> *frame = new VertexAttribute<VEC3, MAP>[3] ;
    if (tangent)
    {
        frame[0] = m_map.template addAttribute<VEC3, VERTEX, MAP>("frameT") ; // Tangent
        attrNames.push_back(frame[0].name()) ;
    }
    if (binormal)
    {
        frame[1] = m_map.template addAttribute<VEC3, VERTEX, MAP>("frameB") ; // Bitangent
        attrNames.push_back(frame[0].name()) ;
    }
    if (normal)
    {
        frame[2] = m_map.template addAttribute<VEC3, VERTEX, MAP>("frameN") ; // Normal
        attrNames.push_back(frame[0].name()) ;
    }

    VertexAttribute<VEC3, MAP> *PBcoefs = NULL, *SHcoefs = NULL ;
    if (PTM)
    {
        PBcoefs = new VertexAttribute<VEC3, MAP>[nbCoefs] ;
        for (unsigned int i = 0 ; i < nbCoefs ; ++i)
        {
            std::stringstream name ;
            name << "PBcoefs" << i ;
            PBcoefs[i] = m_map.template addAttribute<VEC3, VERTEX, MAP>(name.str()) ;
            attrNames.push_back(PBcoefs[i].name()) ;
        }
    }

    if (SH)
    {
        SHcoefs = new VertexAttribute<VEC3, MAP>[nbCoefs] ;
        for (unsigned int i = 0 ; i < nbCoefs ; ++i)
        {
            std::stringstream name ;
            name << "SHcoefs" << i ;
            SHcoefs[i] = m_map.template addAttribute<VEC3, VERTEX, MAP>(name.str()) ;
            attrNames.push_back(SHcoefs[i].name()) ;
        }
    }

    VertexAttribute<REAL, MAP> *remainders = new VertexAttribute<REAL, MAP>[nbRemainders] ;
    for (unsigned int i = 0 ; i < nbRemainders ; ++i)
    {
        std::stringstream name ;
        name << "remainderNo" << i ;
        remainders[i] = m_map.template addAttribute<REAL, VERTEX, MAP>(name.str()) ;
        attrNames.push_back(remainders[i].name()) ;
    }

    // Read vertices
    std::vector<unsigned int> verticesID ;
    verticesID.reserve(nbVertices) ;

    double* properties = new double[nbProps] ;
    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;
    for (unsigned int i = 0 ; i < nbVertices ; ++i) // Read and store properties for current vertex
    {
        unsigned int id = container.insertLine() ;
        verticesID.push_back(id) ;

        fp.read((char*)properties,nbProps * propSize) ;

        // positions
        if (nbProps > 2)
            positions[id] = VEC3(properties[0],properties[1],properties[2]) ; // position

        if (tangent && binormal && normal) // == if (nbprops > 11)
            for (unsigned int k = 0 ; k < 3 ; ++k) // frame
                for (unsigned int l = 0 ; l < 3 ; ++l)
                    frame[k][id][l] = (typename PFP::REAL)(properties[3+(3*k+l)]) ;

        for (unsigned int l = 0 ; l < nbCoefs ; ++l) // coefficients
            for (unsigned int k = 0 ; k < 3 ; ++k)
            {
                if (PTM)
                    PBcoefs[l][id][k] = (typename PFP::REAL)(properties[12+(3*l+k)]) ;
                else /* if SH */
                    SHcoefs[l][id][k] = (typename PFP::REAL)(properties[12+(3*l+k)]) ;
            }

        unsigned int cur = 12+3*nbCoefs ;
        for (unsigned int k = 0 ; k < nbRemainders ; ++k) // remaining data
            remainders[k][id] = (typename PFP::REAL)(properties[cur + k]) ;
    }
    m_nbVertices = verticesID.size() ;
    delete[] properties ;

    // Read faces index
    m_nbEdges.reserve(m_nbFaces) ;
    m_emb.reserve(3 * m_nbFaces) ;
    for (unsigned int i = 0 ; i < m_nbFaces ; ++i)
    {
        // read the indices of vertices for current face
        unsigned int nbEdgesForFace ;
        unsigned char tmp ;
        fp.read((char*)&(tmp), sizeof(unsigned char)) ;
        nbEdgesForFace = tmp ;
        m_nbEdges.push_back(nbEdgesForFace);

        unsigned int vertexID ;
        for (unsigned int j=0 ; j < nbEdgesForFace ; ++j)
        {
            fp.read((char*)&vertexID, sizeof(unsigned int)) ;
            m_emb.push_back(verticesID[vertexID]);
        }
        /*// read the indices of vertices for current face
        unsigned int nbEdgesForFace ;
        fp.read((char*)&(nbEdgesForFace), sizeof(unsigned int)) ;
        m_nbEdges.push_back(nbEdgesForFace);

        unsigned int vertexID ;
        for (unsigned int j=0 ; j < nbEdgesForFace ; ++j)
        {
            fp.read((char*)&vertexID, sizeof(unsigned int)) ;
            m_emb.push_back(verticesID[vertexID]);
        }*/
    }

    // Close file
    fp.close() ;

    return true ;
}

/**
 * Import plyPTM (F Larue format).
 * It handles only 2nd degree polynomials
 * @param filename : the file to import;
 * @param attrNames : reference that will be filled with the attribute names ;
 *  - 1 attrName for geometric position (VEC3)
 *  - 3 attrNames for local frame (3xVEC3) : Tangent, Bitangent and Normal vector
 *  - 6 attrNames for the function coefficients (6xVEC3) : 6 RGB coefficients being successively the quadratic members, the linears and the constants (u then v) : a*u^2 + b*v^2 + c*uv + d*u + e*v +f.
  * @return bool : success.
 */
//template <typename PFP>
//bool MeshTablesSurface<PFP>::importPlyPTM(const std::string& filename, std::vector<std::string>& attrNames)
//{
//	AttributeHandler<typename PFP::VEC3> positions =  m_map.template getAttribute<typename PFP::VEC3>(VERTEX, "position") ;
//
//	if (!positions.isValid())
//		positions = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "position") ;
//
//	attrNames.push_back(positions.name()) ;
//
//	AttributeHandler<typename PFP::VEC3> frame[3] ;
//	frame[0] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "frame_T") ; // Tangent
//	frame[1] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "frame_B") ; // Bitangent
//	frame[2] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "frame_N") ; // Normal
//	for (unsigned int i = 0 ; i < 3 ; ++i)
//		attrNames.push_back(frame[i].name()) ;
//
//	AttributeHandler<typename PFP::VEC3> colorPTM[6] ;
//	colorPTM[0] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "colorPTM_a") ;
//	colorPTM[1] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "colorPTM_b") ;
//	colorPTM[2] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "colorPTM_c") ;
//	colorPTM[3] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "colorPTM_d") ;
//	colorPTM[4] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "colorPTM_e") ;
//	colorPTM[5] = m_map.template addAttribute<typename PFP::VEC3>(VERTEX, "colorPTM_f") ;
//
//	for (unsigned int i = 0 ; i < 6 ; ++i)
//		attrNames.push_back(colorPTM[i].name()) ;
//
//	AttributeContainer& container = m_map.getAttributeContainer(VERTEX) ;
//
//	std::ifstream fp(filename.c_str(), std::ios::binary);
//	if (!fp.good())
//	{
//		CGoGNerr << "Unable to open file " << filename<< CGoGNendl;
//		return false;
//	}
//
//    std::string ligne;
//    std::string tag;
//
//	fp >> tag;
//	if (tag != std::string("ply"))
//	{
//		CGoGNerr <<filename<< " is not a ply file !" <<  CGoGNendl;
//		return false;
//	}
//
//	// va au nombre de sommets
//	do
//	{
//		fp >> tag;
//	} while (tag != std::string("vertex"));
//
//	unsigned int nbp;
//	fp >> nbp;
//	// read points
//	std::vector<unsigned int> verticesID;
//	verticesID.reserve(nbp);
//
//	// va au nombre de faces en comptant le nombre de "property"
//	unsigned int nb_props = 0;
//	do
//	{
//		fp >> tag;
//		if (tag == std::string("property"))
//			nb_props++;
//	} while (tag != std::string("face"));
//
//	fp >> m_nbFaces;
// 	m_nbEdges.reserve(m_nbFaces);
//	m_emb.reserve(3*m_nbFaces);
//
//	// lecture des sommets
//
//	// saute à la fin du header
//	do
//	{
//		fp >> tag;
//	} while (tag != std::string("end_header"));
//
//	float* properties = new float[nb_props];
//
//	for (unsigned int i = 0; i < nbp; ++i)
//	{
//		unsigned int id = container.insertLine();
//		verticesID.push_back(id);
//
//		for (unsigned int j = 0; j < nb_props; ++j)
//		{
//			fp >> properties[j];
//		}
//
//		positions[id] = VEC3(properties[0],properties[1],properties[2]);
//
//		for (unsigned int k = 0 ; k < 3 ; ++k)
//			for (unsigned int l = 0 ; l < 3 ; ++l)
//				frame[k][id][l] = properties[3+(3*k+l)] ;
//
//		for (unsigned int k = 0 ; k < 3 ; ++k)
//			for (unsigned int l = 0 ; l < 6 ; ++l)
//				colorPTM[l][id][k] = properties[12+(6*k+l)];
//	}
//
//	m_nbVertices = verticesID.size();
//	delete[] properties;
//
//// read indices of faces
//	for (unsigned int i = 0; i < m_nbFaces; i++)
//	{
//		// read the indices vertices of face
//		int nbe;
//		fp >> nbe;
//		m_nbEdges.push_back(nbe);
//
//		int pt;
//		for (int j=0; j<nbe; ++j)
//		{
//			fp >> pt;
//			m_emb.push_back(verticesID[pt]);
//		}
//	}
//
//	fp.close();
//	return true;
//}

template <typename PFP>
bool MeshTablesSurface<PFP>::importAHEM(const std::string& filename, std::vector<std::string>& attrNames)
{
    // Open file

    std::ifstream fp(filename.c_str(), std::ios::binary);

    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }

    // Read header

    AHEMHeader hdr;

    fp.read((char*)&hdr, sizeof(AHEMHeader));

    if(hdr.magic != AHEM_MAGIC)
        CGoGNerr << "Warning: " << filename << " invalid magic" << CGoGNendl;

    m_nbVertices = hdr.meshHdr.vxCount;
    m_nbFaces = hdr.meshHdr.faceCount;

    // Read attributes

    AHEMAttributeDescriptor* ahemAttrDesc = new AHEMAttributeDescriptor[hdr.attributesChunkNumber];
    char** ahemAttrNames = new char*[hdr.attributesChunkNumber];

    for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
    {
        fp.read((char*)(ahemAttrDesc + i), sizeof(AHEMAttributeDescriptor));

        ahemAttrNames[i] = new char[ahemAttrDesc[i].nameSize + 1];
        fp.read(ahemAttrNames[i], ahemAttrDesc[i].nameSize);
        ahemAttrNames[i][ahemAttrDesc[i].nameSize] = '\0';
    }

    // Compute buffer size for largest chunk and allocate

    unsigned int bufferSize = hdr.meshHdr.meshChunkSize;

    for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
        if(ahemAttrDesc[i].attributeChunkSize > bufferSize)
            bufferSize = ahemAttrDesc[i].attributeChunkSize;

    char* buffer = new char[bufferSize];

    // Allocate vertices

    AttributeContainer& vxContainer = m_map.template getAttributeContainer<VERTEX>();

    std::vector<unsigned int> verticesId;
    verticesId.resize(hdr.meshHdr.vxCount);

    for(unsigned int i = 0 ; i < hdr.meshHdr.vxCount ; i++)
        verticesId[i] = vxContainer.insertLine();

    // Read faces stream

    m_nbEdges.resize(hdr.meshHdr.faceCount);
    m_emb.resize(hdr.meshHdr.heCount);

    fp.read(buffer, hdr.meshHdr.meshChunkSize);
    char* batch = buffer;

    unsigned int fCount = 0;

    unsigned int fId = 0;
    unsigned int j = 0;

    while(fCount < hdr.meshHdr.faceCount)
    {
        AHEMFaceBatchDescriptor* fbd = (AHEMFaceBatchDescriptor*)batch;
        stUInt32* ix = (stUInt32*)(batch + sizeof(AHEMFaceBatchDescriptor));

        for(unsigned int i = 0 ; i < fbd->batchLength ; i++)
        {
            m_nbEdges[fId++] = fbd->batchFaceSize;

            for(unsigned int k = 0 ; k < fbd->batchFaceSize ; k++)
                m_emb[j++] = *ix++;
        }

        fCount += fbd->batchLength;
        batch = (char*)ix;
    }

    // Read positions

    VertexAttribute<VEC3, MAP> position =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!position.isValid())
        position = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(position.name()) ;

    AHEMAttributeDescriptor* posDesc = NULL;

    for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
    {
        if(IsEqualGUID(ahemAttrDesc[i].semantic, AHEMATTRIBUTE_POSITION))
        {
            posDesc = ahemAttrDesc + i;
            break;
        }
    }

    fp.seekg(posDesc->fileStartOffset, std::ios_base::beg);
    fp.read(buffer, posDesc->attributeChunkSize);

    float* q = (float*)buffer;

    for(unsigned int i = 0 ; i < hdr.meshHdr.vxCount ; i++)
    {
        position[verticesId[i]] = VEC3(q[0], q[1], q[2]);
        q += 3;
    }

    // Close file and release allocated stuff

    fp.close();

    for(unsigned int i = 0 ; i < hdr.attributesChunkNumber ; i++)
        delete[] ahemAttrNames[i];

    delete[] ahemAttrNames;
    delete[] ahemAttrDesc;
    delete[] buffer;

    return true;
}

#ifdef WITH_ASSIMP
//template<typename PFP>
//void MeshTablesSurface<PFP>::extractMeshRec(AttributeContainer& container, VertexAttribute<typename PFP::VEC3>& positions, const struct aiScene* scene, const struct aiNode* nd, struct aiMatrix4x4* trafo)
//{
//    struct aiMatrix4x4 prev;

//    prev = *trafo;
//    aiMultiplyMatrix4(trafo,&nd->mTransformation);

//    std::vector<unsigned int> verticesID;

//    // foreach mesh of node
//    for (unsigned int n = 0; n < nd->mNumMeshes; ++n)
//    {
//        const struct aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];

//        verticesID.clear();
//        verticesID.reserve(mesh->mNumVertices);
//        //read positions
//        for (unsigned int t = 0; t < mesh->mNumVertices; ++t)
//        {
//            // transform position
//            struct aiVector3D tmp = mesh->mVertices[t];
//            aiTransformVecByMatrix4(&tmp, trafo);
//            // now store it
//            unsigned int id = container.insertLine();
//            positions[id] = VEC3(tmp[0], tmp[1], tmp[2]);
//            verticesID.push_back(id);
//        }
//        m_nbVertices += mesh->mNumVertices;

//        // read faces indices
//        for (unsigned int t = 0; t < mesh->mNumFaces; ++t)
//        {
//            const struct aiFace* face = &mesh->mFaces[t];
//            m_nbEdges.push_back(face->mNumIndices);
//            for(unsigned int i = 0; i < face->mNumIndices; i++)
//            {
//                unsigned int pt = face->mIndices[i];
//                m_emb.push_back(verticesID[pt]);
//            }
//        }
//        m_nbFaces += mesh->mNumFaces;
//    }

//    // recurse on all children of node
//    for (unsigned int n = 0; n < nd->mNumChildren; ++n)
//    {
////		CGoGNout << "Children "<<n<< CGoGNendl;
//        extractMeshRec(container, positions, scene, nd->mChildren[n], trafo);
//    }
//    *trafo = prev;
//}


//template <typename PFP>
//bool MeshTablesSurface<PFP>::importASSIMP(const std::string& filename, std::vector<std::string>& attrNames)
//{
//    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;
//    VertexAttribute<typename PFP::VEC3> positions = m_map.template addAttribute<typename PFP::VEC3, VERTEX>("position") ;
//    attrNames.push_back(positions.name()) ;

//    m_nbVertices = 0;
//    m_nbFaces = 0;

//    m_nbEdges.reserve(5000);
//    m_emb.reserve(15000);

//    struct aiMatrix4x4 trafo;
//    aiIdentityMatrix4(&trafo);

//    m_lab = 0;
//    const struct aiScene* scene = aiImportFile(filename.c_str(), aiProcess_FindDegenerates | aiProcess_JoinIdenticalVertices);
//    extractMeshRec(container, positions, scene, scene->mRootNode, &trafo);

//    return true;
//}
#endif

template<typename PFP>
bool MeshTablesSurface<PFP>::mergeCloseVertices()
{
    const int NBV = 64; // seems to be good

    const int NEIGH[27] = {
        -NBV*NBV - NBV - 1, 	-NBV*NBV - NBV, 	-NBV*NBV - NBV + 1,
        -NBV*NBV - 1, 	-NBV*NBV, 	-NBV*NBV + 1,
        -NBV*NBV + NBV - 1,	-NBV*NBV + NBV,	- NBV*NBV + NBV + 1,
        -NBV - 1,	- NBV,	-NBV + 1,
        -1,	0,	1,
        NBV - 1,	NBV,	NBV + 1,
        NBV*NBV - NBV - 1,	NBV*NBV - NBV,	NBV*NBV - NBV + 1,
        NBV*NBV - 1,	NBV*NBV,	NBV*NBV + 1,
        NBV*NBV + NBV - 1,	NBV*NBV + NBV,	NBV*NBV + NBV + 1};

    std::vector<unsigned int>** grid;
    grid = new std::vector<unsigned int>*[NBV*NBV*NBV];

    // init grid with null ptrs
    for (unsigned int i=0; i<NBV*NBV*NBV; ++i)
        grid[i] = NULL;

    VertexAttribute<VEC3, MAP> positions = m_map.template getAttribute<VEC3, VERTEX, MAP>("position");

    // compute BB
    Geom::BoundingBox<typename PFP::VEC3> bb(positions[ positions.begin() ]) ;
    for (unsigned int i = positions.begin(); i != positions.end(); positions.next(i))
    {
        bb.addPoint(positions[i]) ;
    }

    // add one voxel around to avoid testing
    typename PFP::VEC3 bbsize = (bb.max() - bb.min());
    typename PFP::VEC3 one = bbsize/(NBV-2);
    one*= 1.001f;
    bb.addPoint( bb.min() - one);
    bb.addPoint( bb.max() + one);
    bbsize = (bb.max() - bb.min());

    VertexAutoAttribute<unsigned int, MAP> gridIndex(m_map, "gridIndex");
    VertexAutoAttribute<unsigned int, MAP> newIndices(m_map, "newIndices");

    // Store each vertex in the grid and store voxel index in vertex attribute
    for (unsigned int i = positions.begin(); i != positions.end(); positions.next(i))
    {
        typename PFP::VEC3 P = positions[i];
        P -= bb.min();
        float pz = floor((P[2]/bbsize[2])*NBV);
        float py = floor((P[1]/bbsize[1])*NBV);
        float px = floor((P[0]/bbsize[0])*NBV);

        unsigned int index = NBV*NBV*pz + NBV*py + px;

        if (pz==63)
            std::cout << "z 63 bb:"<<bb<<"  P="<<positions[i]<< std::endl;

        std::vector<unsigned int>* vox = grid[index];
        if (vox==NULL)
        {
            grid[index] = new std::vector<unsigned int>();
            grid[index]->reserve(8);
            vox = grid[index];
        }
        vox->push_back(i);
        gridIndex[i] = index;
        newIndices[i] = 0xffffffff;
    }

    // compute EPSILON: average length of 50 of 100 first edges of faces divide by 10000 (very very closed)
    unsigned int nbf = 100;
    if (nbf> m_nbFaces)
        nbf = m_nbFaces;

    int k=0;
    double d=0;
    for (unsigned int i=0; i< nbf;++i)
    {
        typename PFP::VEC3 e1 = positions[m_emb[k+1]] - positions[m_emb[k]];
        d += double(e1.norm());
        k += m_nbEdges[i];
    }
    d /= double(nbf);

    typename PFP::REAL epsilon = d/10000.0;


    // traverse vertices
    for (unsigned int i = positions.begin(); i != positions.end(); positions.next(i))
    {
        if (newIndices[i] == 0xffffffff)
        {
            const typename PFP::VEC3& P = positions[i];

            for (unsigned int n=0; n<27; ++n)
            {
                std::vector<unsigned int>* voxel = grid[gridIndex[i]+NEIGH[n]];
                if (voxel != NULL)
                {
                    for (std::vector<unsigned int>::iterator v = voxel->begin(); v != voxel->end(); ++v)
                    {
                        if ((*v != i) && (*v != 0xffffffff))
                        {
                            typename PFP::VEC3 Q = positions[*v];
                            Q -= P;
                            typename PFP::REAL d2 = Q*Q;
                            if (d2 < epsilon*epsilon)
                            {
                                newIndices[*v] = i;
                                *v = 0xffffffff;
                            }
                        }
                    }
                }
            }
        }
    }

    // update faces indices
    for	(std::vector<unsigned int>::iterator it = m_emb.begin(); it != m_emb.end(); ++it)
    {
        if (newIndices[*it] != 0xffffffff)
        {
            *it = newIndices[*it];
        }
    }

    // delete embeddings
    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    for (unsigned int i = positions.begin(); i != positions.end(); positions.next(i))
    {
        if (newIndices[i] != 0xffffffff)
        {
            container.removeLine(i);
        }
    }

    // release grid memory
    for (unsigned int i=0; i<NBV*NBV*NBV; ++i)
        if (grid[i]!=NULL)
            delete grid[i];

    delete[] grid;

    return true;
}



template<typename PFP>
bool MeshTablesSurface<PFP>::importSTLAscii(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::in);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }


    std::vector<unsigned int> verticesID;
    verticesID.reserve(1000);
    std::vector<VEC3> m_points;

    m_nbFaces = 0;
    m_nbVertices = 0;

    m_nbEdges.reserve(1000);
    m_emb.reserve(3000);

    std::string ligne;
    std::string tag;

    do
    {
        fp >> tag;
        std::getline (fp, ligne);
    }while (tag != std::string("vertex"));

    do
    {
        for (int i=0; i<3; ++i)
        {
            std::stringstream oss(ligne);
            float x,y,z;
            oss >> x;
            oss >> y;
            oss >> z;

            VEC3 pos(x,y,z);

            typename std::vector<VEC3>::iterator it = find (m_points.begin(), m_points.end(), pos);
            if (it != m_points.end())
            {
                unsigned int idP = it - m_points.begin();
                m_emb.push_back(verticesID[idP]);
            }
            else
            {
                unsigned int id = container.insertLine();
                positions[id] = pos;
                verticesID.push_back(id);
                m_points.push_back(pos);
                m_emb.push_back(id);
            }
            fp >> tag;
            std::getline (fp, ligne);
        }
        m_nbEdges.push_back(3);

        do
        {
            fp >> tag;
            std::getline (fp, ligne);
        }while (!fp.eof() && (tag != std::string("vertex")));

    }while (!fp.eof());

    m_nbVertices = m_points.size();
    m_nbFaces = m_nbEdges.size();

    fp.close();
    return true;
}



template<typename PFP>
bool MeshTablesSurface<PFP>::importSTLBin(const std::string& filename, std::vector<std::string>& attrNames)
{
    VertexAttribute<VEC3, MAP> positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;

    if (!positions.isValid())
        positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;

    attrNames.push_back(positions.name()) ;

    AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

    // open file
    std::ifstream fp(filename.c_str(), std::ios::in|std::ios::binary);
    if (!fp.good())
    {
        CGoGNerr << "Unable to open file " << filename << CGoGNendl;
        return false;
    }


    std::vector<unsigned int> verticesID;
    verticesID.reserve(1000);
    std::vector<VEC3> m_points;

    m_nbVertices = 0;


    std::string ligne;
    std::string tag;

    fp.ignore(80);
    fp.read(reinterpret_cast<char*>(&m_nbFaces), sizeof(int));

    m_nbEdges.reserve(m_nbFaces);
    m_emb.reserve(m_nbFaces*3);


    for (unsigned int j=0;j<m_nbFaces;++j)
    {
        fp.ignore(3*sizeof(float));
        for (int i=0; i<3; ++i)
        {
            VEC3 pos;
            fp.read(reinterpret_cast<char*>(&pos[0]), sizeof(float));
            fp.read(reinterpret_cast<char*>(&pos[1]), sizeof(float));
            fp.read(reinterpret_cast<char*>(&pos[2]), sizeof(float));

            typename std::vector<VEC3>::iterator it = find (m_points.begin(), m_points.end(), pos);
            if (it != m_points.end())
            {
                unsigned int idP = it - m_points.begin();
                m_emb.push_back(verticesID[idP]);
            }
            else
            {
                unsigned int id = container.insertLine();
                positions[id] = pos;
                verticesID.push_back(id);
                m_points.push_back(pos);
                m_emb.push_back(id);
            }
        }
        fp.ignore(2);
        m_nbEdges.push_back(3);
    }

    m_nbVertices = m_points.size();
    m_nbFaces = m_nbEdges.size();

    fp.close();
    return true;
}

} // namespace Import

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
