/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/io/MeshSTL.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/logging/Messaging.h>

/// This line register the MeshSTL to the messaging system so that we
/// can use the msg_info() instead of msg_info("MeshSTL").
MSG_REGISTER_CLASS(sofa::helper::io::MeshSTL, "MeshSTL")

#ifndef NDEBUG
#define EMIT_DEBUG_MESSAGE true
#else
#define EMIT_DEBUG_MESSAGE false
#endif

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

SOFA_DECL_CLASS(MeshSTL)
Creator<Mesh::FactoryMesh,MeshSTL> MeshSTLClass("stl");

void MeshSTL::init (std::string filename)
{
    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error() << "File " << filename << " not found ";
        return;
    }
    loaderType = "stl";
    std::ifstream file(filename.c_str());

    if (!file.good())
    {
       file.close();
       msg_error() << "Cannot read file '" << filename << "'.";
       return;
    }

    std::size_t namepos = filename.find_last_of("/");
    std::string name = filename.substr(namepos+1);

    std::string token;
    file >> token;
    if (token == "solid")
    {
        dmsg_info_when(EMIT_DEBUG_MESSAGE) << "Reading STL file : " << name;
        readSTL(file);
    }
    else
    {
        dmsg_info_when(EMIT_DEBUG_MESSAGE) <<  "Reading binary STL file : " << name;
        file.close();
        readBinarySTL(filename);
    }

    // announce the model statistics
    dmsg_info_when(EMIT_DEBUG_MESSAGE) << " Vertices: " << vertices.size() << msgendl
                                       << " Normals: " << normals.size() << msgendl
                                       << " Texcoords: " << texCoords.size() << msgendl
                                       << " Triangles: " << facets.size() ;

    if (vertices.size()>0)
    {
        // compute bbox
        Vector3 minBB = vertices[0];
        Vector3 maxBB = vertices[0];
        for (unsigned int i=1; i<vertices.size(); i++)
        {
            Vector3 p = vertices[i];
            for (int c=0; c<3; c++)
            {
                if (minBB[c] > p[c])
                    minBB[c] = p[c];
                if (maxBB[c] < p[c])
                    maxBB[c] = p[c];
            }
        }

       msg_info_when(EMIT_DEBUG_MESSAGE) << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">";
    }

}


void MeshSTL::readSTL(std::ifstream &file)
{
    /* http://www.ennex.com/~fabbers/StL.asp */

    Vec3f result;

    std::string line;
    std::map< defaulttype::Vec3f, unsigned > map;
    unsigned positionCounter = 0u, vertexCounter=0u;

    // there must be a way to perform it at compile time with initializer_list
    vector< vector<int> > vertNormTexIndices(5); // 3 pos + 1 normal + 1 texcoord
    {
        vector<int> nullIndices(3);
        std::fill( nullIndices.begin(), nullIndices.end(), 0 );
        std::fill( vertNormTexIndices.begin(), vertNormTexIndices.end(), nullIndices );
    }

    while( std::getline(file,line) )
    {
        if (line.empty()) continue;
        std::istringstream values(line);
        std::string token;
        values >> token;
        if (token == "facet")
        {
            /* normal */
            values >> token >> result[0] >> result[1] >> result[2];
            //normals.push_back(result);
        }
        else if (token == "vertex")
        {
            /* vertex */
            values >> result[0] >> result[1] >> result[2];

            auto it = map.find( result );
            if( it == map.end() )
            {
                vertNormTexIndices[0][vertexCounter++] = positionCounter;
                map[result] = positionCounter++;
                vertices.push_back(result);
            }
            else
            {
                vertNormTexIndices[0][vertexCounter++] = it->second;
            }

        }
        else if (token == "endfacet")
        {
            facets.push_back(vertNormTexIndices);
            vertexCounter=0;
        }
        else if (token == "endsolid" || token == "end")
            break;
    }

    file.close();

}

void MeshSTL::readBinarySTL (const std::string &filename)
{
    /* Based on MeshSTLLoader */
    /* http://www.ennex.com/~fabbers/StL.asp */

    std::ifstream dataFile(filename.c_str(), std::ios::in | std::ios::binary);


    Vec3f result;
    unsigned int attributeCount;

    std::map< defaulttype::Vec3f, unsigned > map;
    unsigned positionCounter = 0u;


    // Skipping header file
    char buffer[80];
    dataFile.read(buffer, 80);

    // Get number of facets
    uint32_t nbrFacet;
    dataFile.read((char*)&nbrFacet, 4);

    // preallocating facets
    // there must be a way to perform part of it at compile time with initializer_list
    {
        vector<int> nullIndices(3);
        std::fill( nullIndices.begin(), nullIndices.end(), 0 );
        vector<vector<int>> vertNormTexIndices(5); // 3 pos + 1 normal + 1 texcoord
        std::fill( vertNormTexIndices.begin(), vertNormTexIndices.end(), nullIndices );
        facets.resize(nbrFacet);
        std::fill( facets.begin(), facets.end(), vertNormTexIndices );
    }


#ifndef NDEBUG
    {
    // checking that the file is large enough to contain the given nb of facets
    // store current position in file
    std::streampos pos = dataFile.tellg();
    // get length of file
    dataFile.seekg(0, std::ios::end);
    std::streampos length = dataFile.tellg();
    // restore pos
    dataFile.seekg(pos);
    // check length
    assert( length >= (std::streampos)( 80 /*header*/ + 4 /*nb facets*/ + nbrFacet * (12 /*normal*/ + 3 * 12 /*points*/ + 2 /*attribute*/ ) ) );
    }
#endif


    // Parsing facets
    for (unsigned int i = 0; i<nbrFacet; ++i)
    {
        // Get normal
        dataFile.read((char*)&result[0], 4);
        dataFile.read((char*)&result[1], 4);
        dataFile.read((char*)&result[2], 4);

        vector< vector<int> >& facet = facets[i];

        // Get vertex
        for (unsigned int j = 0; j<3; ++j)
        {
            dataFile.read((char*)&result[0], 4);
            dataFile.read((char*)&result[1], 4);
            dataFile.read((char*)&result[2], 4);

            auto it = map.find( result );
            if( it == map.end() )
            {
                facet[0][j] = positionCounter;
                map[result] = positionCounter++;
                vertices.push_back(result);
            }
            else
            {
                facet[0][j] = it->second;
            }
        }

        // Attribute byte count
        dataFile.read((char*)&attributeCount, 2);
    }
}


} // namespace io

} // namespace helper

} // namespace sofa

