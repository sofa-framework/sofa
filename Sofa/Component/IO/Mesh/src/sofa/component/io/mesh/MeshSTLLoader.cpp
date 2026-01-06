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
#include <sofa/component/io/mesh/MeshSTLLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <string>

namespace sofa::component::io::mesh
{

using sofa::helper::getWriteOnlyAccessor;

using namespace sofa::type;
using namespace sofa::defaulttype;

void registerMeshSTLLoader(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Loader for the STL file format. STL can be used to represent the surface of object using with a triangulation.")
        .add< MeshSTLLoader >());
}

//Base VTK Loader
MeshSTLLoader::MeshSTLLoader() : MeshLoader()
    , d_headerSize(initData(&d_headerSize, 80u, "headerSize", "Size of the header binary file (just before the number of facet)."))
    , d_forceBinary(initData(&d_forceBinary, false, "forceBinary", "Force reading in binary mode. Even in first keyword of the file is solid."))
    , d_mergePositionUsingMap(initData(&d_mergePositionUsingMap, true, "mergePositionUsingMap","Since positions are duplicated in a STL, they have to be merged. Using a map to do so will temporarily duplicate memory but should be more efficient. Disable it if memory is really an issue."))
{
}


bool MeshSTLLoader::doLoad()
{
    const char* filename = d_filename.getFullPath().c_str();
    std::string sfilename(filename);
    if (!sofa::helper::system::DataRepository.findFile(sfilename))
    {
        msg_error(this) << "File " << filename << " not found ";
        return false;
    }

    std::ifstream file(filename);
    if (!file.good())
    {
        file.close();
        msg_error(this) << "Cannot read file '" << filename << "'.";
        return false;
    }

    bool ret = false;
    if( d_forceBinary.getValue() )
        ret = this->readBinarySTL(filename); // -- Reading binary file
    else
    {
        std::string test;
        file >> test;

        if ( test == "solid" )
            ret = this->readSTL(file);
        else
        {
            file.close(); // no longer need for an ascii-open file
            ret = this->readBinarySTL(filename); // -- Reading binary file
        }
    }
    return ret;
}

bool isBinarySTLValid(const char* filename, const MeshSTLLoader* _this)
{
    // Binary STL files have 80-bytes headers. The following 4-bytes is the number of triangular d_facets in the file
    // Each facet is described with a 50-bytes field, so a valid binary STL file verifies the following condition:
    // nFacets * 50 + 84-bytes header == filesize

    long filesize;
    std::ifstream f(filename, std::ifstream::ate | std::ifstream::binary);
    filesize = f.tellg();
    if (filesize < 84)
    {
        msg_error(_this) << "Can't read binary STL file: " << filename;
        return false;
    }
    f.seekg(0);
    char buffer[80];
    f.read(buffer, 80);
    uint32_t ntriangles;
    f.read(reinterpret_cast<char*>(&ntriangles), 4);
    const uint32_t expectedFileSize = ntriangles * 50 + 84;
    if (filesize != expectedFileSize)
    {
        msg_error(_this) << filename << " isn't binary STL file. File size expected to be "
            << expectedFileSize << " (with " << ntriangles << " triangles) but it is " << filesize;
        return false;
    }
    return true;
}

bool MeshSTLLoader::readBinarySTL(const char *filename)
{
    dmsg_info() << "Reading binary STL file..." ;
    if (!isBinarySTLValid(filename, this))
        return false;

    auto my_positions = getWriteOnlyAccessor(d_positions);
    auto my_normals = getWriteOnlyAccessor(d_normals);
    auto my_triangles = getWriteOnlyAccessor(this->d_triangles);

    std::map< sofa::type::Vec3, sofa::Index > my_map;
    sofa::Index positionCounter = 0;
    const bool useMap = d_mergePositionUsingMap.getValue();

    std::ifstream dataFile(filename, std::ios::in | std::ifstream::binary);

    // Skipping header file
    char buffer[256];
    dataFile.read(buffer, d_headerSize.getValue());

    uint32_t nbrFacet;
    dataFile.read(reinterpret_cast<char*>(&nbrFacet), 4);

    my_normals.resize( nbrFacet ); // exact size
    my_positions.reserve( nbrFacet * 3 ); // max size

#ifndef NDEBUG
    {
    // checking that the file is large enough to contain the given nb of d_facets
    // store current pos in file
    std::streampos pos = dataFile.tellg();
    // get length of file
    dataFile.seekg(0, std::ios::end);
    std::streampos length = dataFile.tellg();
    // restore pos in file
    dataFile.seekg(pos);
    // check for length
    assert(length >= d_headerSize.getValue() + 4 + nbrFacet * (12 /*normal*/ + 3 * 12 /*points*/ + 2 /*attribute*/ ) );
    }
#endif

    // temporaries
    sofa::type::Vec3f vertexf, normalf;

    // reserve vector before filling it
    my_triangles.reserve( nbrFacet );

    unsigned int nbDegeneratedTriangles = 0;

    // Parsing d_facets
    for (uint32_t i = 0; i<nbrFacet; ++i)
    {
        topology::Triangle the_tri;

        // Normal:
        dataFile.read((char*)&normalf[0], 4);
        dataFile.read((char*)&normalf[1], 4);
        dataFile.read((char*)&normalf[2], 4);
        my_normals[i] = type::toVec3(normalf);

        // Vertices:
        for (size_t j = 0; j<3; ++j)
        {
            dataFile.read((char*)&vertexf[0], 4);
            dataFile.read((char*)&vertexf[1], 4);
            dataFile.read((char*)&vertexf[2], 4);
            const auto vertex = type::toVec3(vertexf);


            if( useMap )
            {
                auto it = my_map.find( vertex );
                if( it == my_map.end() )
                {
                    the_tri[j] = positionCounter;
                    my_map[vertex] = positionCounter++;
                    my_positions.push_back(vertex);
                }
                else
                {
                    the_tri[j] = it->second;
                }
            }
            else
            {
                bool find = false;
                for (size_t k=0; k<my_positions.size(); ++k)
                    if ( (vertex[0] == my_positions[k][0]) && (vertex[1] == my_positions[k][1])  && (vertex[2] == my_positions[k][2]))
                    {
                        find = true;
                        the_tri[j] = static_cast<sofa::Index>(k);
                        break;
                    }

                if (!find)
                {
                    my_positions.push_back(vertex);
                    the_tri[j] = my_positions.size()-1;
                }
            }
        }

        if (the_tri[0] == the_tri[1] || the_tri[1] == the_tri[2] || the_tri[0] == the_tri[2])
        {
            ++nbDegeneratedTriangles;
        }
        else
        {
            this->addTriangle(my_triangles.wref(), the_tri);
        }

        // Attribute byte count
        uint16_t count;
        dataFile.read((char*)&count, 2);
    }

    if(my_triangles.size() != (size_t)(nbrFacet - nbDegeneratedTriangles))
    {
        msg_error() << "Size mismatch between triangle vector and facetSize";
        return false;
    }

    msg_warning_when(nbDegeneratedTriangles > 0) << "Found " << nbDegeneratedTriangles << " degenerated triangles ("
        "triangles which indices are not all different). Those triangles have not been added to the list of triangles";

    dmsg_info() << "done!" ;
    return true;
}


bool MeshSTLLoader::readSTL(std::ifstream& dataFile)
{
    Vec3 result;
    std::string line;

    auto my_positions = getWriteOnlyAccessor(d_positions);
    auto my_normals = getWriteOnlyAccessor(d_normals);
    auto my_triangles = getWriteOnlyAccessor(d_triangles);

    std::map< sofa::type::Vec3, sofa::Index > my_map;
    sofa::Index positionCounter = 0, vertexCounter = 0;
    const bool useMap = d_mergePositionUsingMap.getValue();

    topology::Triangle the_tri;

    while (std::getline(dataFile, line))
    {
        if (line.empty()) continue;
        std::istringstream values(line);

        std::string bufferWord;
        values >> bufferWord;

        if (bufferWord == "facet")
        {
            // Normal
            values >> bufferWord >> result[0] >> result[1] >> result[2];
            my_normals.push_back(result);
        }
        else if (bufferWord == "vertex")
        {
            // Vertex
            values >> result[0] >> result[1] >> result[2];

            if( useMap )
            {
                auto it = my_map.find(result);
                if( it == my_map.end() )
                {
                    the_tri[vertexCounter] = positionCounter;
                    my_map[result] = positionCounter++;
                    my_positions.push_back(result);
                }
                else
                {
                    the_tri[vertexCounter] = it->second;
                }
            }
            else
            {

                bool find = false;
                for (size_t i=0; i<my_positions.size(); ++i)
                    if ( (result[0] == my_positions[i][0]) && (result[1] == my_positions[i][1])  && (result[2] == my_positions[i][2]))
                    {
                        find = true;
                        the_tri[vertexCounter] = static_cast<sofa::Index>(i);
                        break;
                    }

                if (!find)
                {
                    my_positions.push_back(result);
                    the_tri[vertexCounter] = static_cast<sofa::Index>(my_positions.size()-1);
                }
            }
            vertexCounter++;
        }
        else if (bufferWord == "endfacet")
        {
            this->addTriangle(my_triangles.wref(), the_tri);
            vertexCounter = 0;
        }
        else if (bufferWord == "endsolid" || bufferWord == "end")
        {
            break;
        }
    }

    dataFile.close();

    dmsg_info() << "done!" ;

    return true;
}

void MeshSTLLoader::doClearBuffers()
{
    /// Nothing to do if no output is added to the "filename" dataTrackerEngine.
}

} //namespace sofa::component::io::mesh
