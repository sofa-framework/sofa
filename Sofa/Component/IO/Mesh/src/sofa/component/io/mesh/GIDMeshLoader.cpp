/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <algorithm>
#include <string>
#include <fstream>

#include <sofa/component/io/mesh/GIDMeshLoader.h>
#include <sofa/core/ObjectFactory.h>

using namespace sofa::helper;

namespace sofa::component::io::mesh
{

void registerGIDMeshLoader(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Load volumetric meshes generated by GID. Some element types are not implemented.")
        .add< GIDMeshLoader >());
}

GIDMeshLoader::GIDMeshLoader() :
    MeshLoader()
{
}

GIDMeshLoader::~GIDMeshLoader()
{
}

bool GIDMeshLoader::doLoad()
{
    std::ifstream file(d_filename.getFullPath().c_str());

    if( !file.good() )
    {
        msg_error() << "Unable to open file " << d_filename.getFullPath();
        return false;
    }

    return readGID(file);
}

bool GIDMeshLoader::readGID(std::ifstream &file)
{
    auto vertices = getWriteOnlyAccessor(d_positions);

    std::string line;
    std::istringstream iss;
    std::getline(file, line);

    iss.str(line);
    std::string finput;

    iss >> finput;

    msg_info() << finput ;
    if( finput == "MESH" )
    {
        if( iss.eof() )
        {
            msg_error() << "Bad GID file";
            return false;
        }

        iss >> finput;

        while( finput != "dimension" )
            iss >> finput;

        if( iss.eof() )
        {
            msg_error() << "Bad GID file";
            return false;
        }

        if( finput == "dimension" )
        {

            if( !(iss >> m_dimensions) )
            {
                msg_error() << "Bah GID mesh header : missing dimension information";
                return false;
            }

            msg_info() << "dimensions = " << m_dimensions;

            if( iss.eof() )
            {
                msg_error() << "Bad GID file";
                return false;
            }

            iss >> finput;

            if( finput == "ElemType" )
            {
                std::string element = "";
                iss >> element;

                if( element == "Linear" )
                    m_eltType = LINEAR;

                if( element == "Triangle" )
                    m_eltType = TRIANGLE;

                if( element == "Quadrilateral" )
                    m_eltType = QUADRILATERAL;

                if( element == "Tetrahedra" )
                    m_eltType = TETRAHEDRA;

                if( element == "Hexahedra" )
                    m_eltType = HEXAHEDRA;

                msg_info() << "Elemtype = " << element;

                if( element == "Prism" )
                {
                    msg_error() << "Element type Prism is currently unsupported in SOFA";
                    return false;
                }

                if( element == "Pyramid" )
                {
                    msg_error() << "Element type Pyramid is currently unsupported in SOFA";
                    return false;
                }

                if( element == "Sphere" )
                {
                    msg_error() << "Element type Sphere is currently unsupported in SOFA";
                    return false;
                }

                if( element == "Circle" )
                {
                    msg_error() << "Element type Circle is currently unsupported in SOFA";
                    return false;
                }

                if( element.empty() )
                {
                    msg_error() << "Bad GID file header : unknown element type.";
                    return false;
                }

                iss >> finput;

                if( finput == "Nnode" )
                {
                    if( !(iss >> m_nNode) )
                    {
                        msg_error() << "Bad GID file header";
                        return false;
                    }

                    msg_info() << "Nnodes = " << m_nNode;

                    if( (m_eltType == LINEAR) && ( m_nNode != 2 && m_nNode != 3 ) )
                    {
                        msg_error() << "Incompatible node count for element type Linear : expected 2 or 3, found " << m_nNode << ".";
                        return false;
                    }

                    if( (m_eltType == TRIANGLE) && ( m_nNode != 3 && m_nNode != 6 ) )
                    {
                        msg_error() << "Incompatible node count for element type Triangle : expected 2 or 3, found " << m_nNode << ".";
                        return false;
                    }

                    if( (m_eltType == QUADRILATERAL) && ( m_nNode != 4 && m_nNode != 8 && m_nNode != 9 ) )
                    {
                        msg_error() << "Incompatible node count for element type Quadrilateral : expected 4, 8 or 9, found " << m_nNode << ".";
                        return false;
                    }


                    if( (m_eltType == TETRAHEDRA) && ( m_nNode != 4 && m_nNode != 10 ) )
                    {
                        msg_error() << "Incompatible node count for element type Tetrahedra : expected 4 or 10, found " << m_nNode << ".";
                        return false;
                    }

                    if( (m_eltType == HEXAHEDRA) && ( m_nNode != 8 && m_nNode != 20 && m_nNode != 27 ) )
                    {
                        msg_error() << "Incompatible node count for element type Quadrilateral : expected 8, 20 or 27, found " << m_nNode << ".";
                        return false;
                    }

                }
                else
                {
                    msg_error() << "Bad GID file header : expecting Nnode tag.";
                    return false;
                }

            }
            else
            {
                msg_error() << "Bad GID file header : missing Elemtype tag.";
                return false;
            }
        }
        else
        {
            msg_error() << "Bad GID file header : missing dimension tag.";
            return false;
        }
    }
    else
    {
        msg_error() << "Bad GID mesh header";
        return false;
    }

    do
    {
        std::getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }
    while( line != "coordinates" );

    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);


    while( line != "end coordinates")
    {
        int vid;


        iss.str(line);
        iss.clear();
        iss >> vid; // vertex id

        Coord vtx;
        for(unsigned char d = 0 ; d < m_dimensions ; ++d)
        {
            iss >> vtx(d);
        }

        vertices.push_back(vtx);

        if( !file.good() )
        {
            msg_error() << "Bad GID file : unexpected EOF.";
            return false;
        }

        getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }

    do
    {
        std::getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }
    while( line != "elements" );

    switch(m_eltType)
    {
    case LINEAR :
        return readLinearElements(file);
        break;

    case TRIANGLE :
        return readTriangleElements(file);
        break;

    case QUADRILATERAL :
        return readQuadrilateralElements(file);
        break;

    case TETRAHEDRA :
        return readTetrahedralElements(file);
        break;

    case HEXAHEDRA :
        return readHexahedralElements(file);
        break;

    default :
        break;
    }

    return false;
}

bool GIDMeshLoader::readLinearElements(std::ifstream &file)
{
    std::string line;
    std::istringstream iss;
    auto meshEdges = getWriteOnlyAccessor(d_edges);

    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);

    if( m_nNode != 2 )
        msg_warning() << "Implementation only supports 2 nodes Linear elements";

    while( line != "end elements")
    {
        unsigned int eid; // element id
        unsigned int vid; // vertex id
        iss.str(line);
        iss.clear();

        iss >> eid;

        Edge e;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        e[0] = vid - 1;


        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        e[1] = vid - 1;

        // we do not treat the middle vertex in the case of a 3 node Linear element

        meshEdges.push_back(e);

        if( !file.good() )
        {
            msg_error() << "Bad GID file : unexpected EOF.";
            return false;
        }

        getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }

    return true;
}

bool GIDMeshLoader::readTriangleElements(std::ifstream &file)
{
    std::string line;
    std::istringstream iss;
    auto meshTriangles = getWriteOnlyAccessor(d_triangles);

    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);

    if( m_nNode != 3 )
        msg_warning() << "Implementation only supports 3 nodes Triangle elements";

    while( line != "end elements")
    {
        unsigned int eid; // element id
        unsigned int vid; // vertex id
        iss.str(line);
        iss.clear();

        iss >> eid;

        Triangle t;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        t[0] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        t[1] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        t[2] = vid - 1;

        meshTriangles.push_back(t);

        if( !file.good() )
        {
            msg_error() << "Bad GID file : unexpected EOF.";
            return false;
        }

        getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }

    return true;
}

bool GIDMeshLoader::readQuadrilateralElements(std::ifstream &file)
{
    std::string line;
    std::istringstream iss;
    auto meshQuads = getWriteOnlyAccessor(d_quads);

    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);

    if( m_nNode != 4 )
        msg_warning() << "Implementation only supports 4 nodes Quadrilateral elements";

    while( line != "end elements")
    {
        unsigned int eid; // element id
        unsigned int vid; // vertex id
        iss.str(line);
        iss.clear();

        iss >> eid;

        Quad q;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        q[0] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        q[1] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        q[2] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        q[3] = vid - 1;


        meshQuads.push_back(q);

        if( !file.good() )
        {
            msg_error() << "Bad GID file : unexpected EOF.";
            return false;
        }

        getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }

    return true;
}

bool GIDMeshLoader::readTetrahedralElements(std::ifstream &file)
{
    std::string line;
    std::istringstream iss;
    auto meshTetra = getWriteOnlyAccessor(d_tetrahedra);

    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);

    if( m_nNode != 4 )
        msg_warning() << "Implementation only supports 4 nodes Tetrahedra elements";

    while( line != "end elements")
    {
        unsigned int eid; // element id
        unsigned int vid; // vertex id
        iss.str(line);
        iss.clear();

        iss >> eid;

        Tetrahedron t;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        t[0] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        t[1] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        t[2] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        t[3] = vid - 1;


        meshTetra.push_back(t);

        if( !file.good() )
        {
            msg_error() << "Bad GID file : unexpected EOF.";
            return false;
        }

        getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }

    return true;
}

bool GIDMeshLoader::readHexahedralElements(std::ifstream &file)
{
    std::string line;
    std::istringstream iss;
    auto meshHexa = getWriteOnlyAccessor(d_hexahedra);

    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);

    if( m_nNode != 8 )
        msg_warning() << "Implementation only supports 8 nodes Hexahedra elements";

    while( line != "end elements")
    {
        unsigned int eid; // element id
        unsigned int vid; // vertex id
        iss.str(line);
        iss.clear();

        iss >> eid;

        Hexahedron h;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[0] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[1] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[2] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[3] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[4] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[5] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[6] = vid - 1;

        if( !(iss >> vid) )
        {
            msg_error() << "Reading GID file : expecting node index";
            return false;
        }
        h[7] = vid - 1;


        meshHexa.push_back(h);

        if( !file.good() )
        {
            msg_error() << "Bad GID file : unexpected EOF.";
            return false;
        }

        getline(file, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    }

    return true;
}


void GIDMeshLoader::doClearBuffers()
{
    m_dimensions = 0;
    m_nNode = 0;
}

} //namespace sofa::component::io::mesh
