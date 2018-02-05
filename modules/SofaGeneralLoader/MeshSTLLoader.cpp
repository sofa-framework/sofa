/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/ObjectFactory.h>
#include <SofaGeneralLoader/MeshSTLLoader.h>
#include <sofa/core/visual/VisualParams.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <string>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshSTLLoader)

int MeshSTLLoaderClass = core::RegisterObject("Specific mesh loader for STL file format.")
        .add< MeshSTLLoader >()
        ;

//Base VTK Loader
MeshSTLLoader::MeshSTLLoader() : MeshLoader()
    , _headerSize(initData(&_headerSize, 80u, "headerSize","Size of the header binary file (just before the number of facet)."))
    , _forceBinary(initData(&_forceBinary, false, "forceBinary","Force reading in binary mode. Even in first keyword of the file is solid."))
    , d_mergePositionUsingMap(initData(&d_mergePositionUsingMap, true, "mergePositionUsingMap","Since positions are duplicated in a STL, they have to be merged. Using a map to do so will temporarily duplicate memory but should be more efficient. Disable it if memory is really an issue."))
{
}



bool MeshSTLLoader::load()
{
    const char* filename = m_filename.getFullPath().c_str();
    std::ifstream file(filename);
    if (!file.good())
    {
        file.close();
        serr << "Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    if( _forceBinary.getValue() )
        return this->readBinarySTL(filename); // -- Reading binary file

    std::string test;
    file >> test;

    if ( test == "solid" )
        return this->readSTL(file);
    else
    {
        file.close(); // no longer need for an ascii-open file
        return this->readBinarySTL(filename); // -- Reading binary file
    }
}


bool MeshSTLLoader::readBinarySTL(const char *filename)
{
    dmsg_info() << "Reading binary STL file..." ;

    std::ifstream dataFile (filename, std::ios::in | std::ios::binary);

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(this->d_positions.beginWriteOnly());
    helper::vector<sofa::defaulttype::Vector3>& my_normals = *(this->d_normals.beginWriteOnly());
    helper::vector<Triangle >& my_triangles = *(this->d_triangles.beginWriteOnly());

    std::map< sofa::defaulttype::Vec3f, core::topology::Topology::index_type > my_map;
    core::topology::Topology::index_type positionCounter = 0;
    bool useMap = d_mergePositionUsingMap.getValue();



    // Skipping header file
    char buffer[256];
    dataFile.read(buffer, _headerSize.getValue());
//    sout << "Header binary file: "<< buffer << sendl;

    uint32_t nbrFacet;
    dataFile.read((char*)&nbrFacet, 4);

    my_triangles.resize( nbrFacet ); // exact size
    my_normals.resize( nbrFacet ); // exact size
    my_positions.reserve( nbrFacet * 3 ); // max size

#ifndef NDEBUG
    {
    // checking that the file is large enough to contain the given nb of facets
    // store current pos in file
    std::streampos pos = dataFile.tellg();
    // get length of file
    dataFile.seekg(0, std::ios::end);
    std::streampos length = dataFile.tellg();
    // restore pos in file
    dataFile.seekg(pos);
    // check for length
    assert( length >= _headerSize.getValue() + 4 + nbrFacet * (12 /*normal*/ + 3 * 12 /*points*/ + 2 /*attribute*/ ) );
    }
#endif

    // temporaries
    sofa::defaulttype::Vec3f vertex, normal;

    // Parsing facets
    for (uint32_t i = 0; i<nbrFacet; ++i)
    {
        Triangle& the_tri = my_triangles[i];

        // Normal:
        dataFile.read((char*)&normal[0], 4);
        dataFile.read((char*)&normal[1], 4);
        dataFile.read((char*)&normal[2], 4);
        my_normals[i] = normal;

        // Vertices:
        for (size_t j = 0; j<3; ++j)
        {
            dataFile.read((char*)&vertex[0], 4);
            dataFile.read((char*)&vertex[1], 4);
            dataFile.read((char*)&vertex[2], 4);


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
                        the_tri[j] = static_cast<core::topology::Topology::PointID>(k);
                        break;
                    }

                if (!find)
                {
                    my_positions.push_back(vertex);
                    the_tri[j] = my_positions.size()-1;
                }
            }

        }

        // Attribute byte count
        uint16_t count;
        dataFile.read((char*)&count, 2);

        // Security: // checked once before reading in debug mode
//        position = dataFile.tellg();
//        if (position == length)
//            break;
    }

    this->d_positions.endEdit();
    this->d_triangles.endEdit();
    this->d_normals.endEdit();

    dmsg_info() << "done!" ;

    return true;
}


bool MeshSTLLoader::readSTL(std::ifstream& dataFile)
{
    dmsg_info() << "Reading STL file..." ;

    // Init
    std::string buffer;
    std::string name; // name of the solid, needed?

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(d_positions.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& my_normals = *(d_normals.beginEdit());
    helper::vector<Triangle >& my_triangles = *(d_triangles.beginEdit());


    std::map< sofa::defaulttype::Vec3f, core::topology::Topology::index_type > my_map;
    core::topology::Topology::index_type positionCounter = 0;
    bool useMap = d_mergePositionUsingMap.getValue();


    // get length of file:
    dataFile.seekg(0, std::ios::end);
    std::streampos length = dataFile.tellg();
    dataFile.seekg(0, std::ios::beg);

    // Reading header
    dataFile >> buffer >> name;

    Triangle the_tri;
    size_t cpt = 0;
    std::streampos position = 0;

    // Parsing facets
    while (position < length)
    {
        sofa::defaulttype::Vector3 normal, vertex;

        std::getline(dataFile, buffer);
        std::stringstream line;
        line << buffer;

        std::string bufferWord;
        line >> bufferWord;

        if (bufferWord == "facet")
        {
            line >> bufferWord >> normal[0] >> normal[1] >> normal[2];
            my_normals.push_back(normal);
        }
        else if (bufferWord == "vertex")
        {
            line >> vertex[0] >> vertex[1] >> vertex[2];

            if( useMap )
            {
                auto it = my_map.find( vertex );
                if( it == my_map.end() )
                {
                    the_tri[cpt] = positionCounter;
                    my_map[vertex] = positionCounter++;
                    my_positions.push_back(vertex);
                }
                else
                {
                    the_tri[cpt] = it->second;
                }
            }
            else
            {

                bool find = false;
                for (size_t i=0; i<my_positions.size(); ++i)
                    if ( (vertex[0] == my_positions[i][0]) && (vertex[1] == my_positions[i][1])  && (vertex[2] == my_positions[i][2]))
                    {
                        find = true;
                        the_tri[cpt] = static_cast<core::topology::Topology::PointID>(i);
                        break;
                    }

                if (!find)
                {
                    my_positions.push_back(vertex);
                    the_tri[cpt] = static_cast<core::topology::Topology::PointID>(my_positions.size()-1);
                }
            }
            cpt++;
        }
        else if (bufferWord == "endfacet")
        {
            my_triangles.push_back(the_tri);
            cpt = 0;
        }
        else if (bufferWord == "endsolid" || bufferWord == "end")
        {
            break;
        }

        position = dataFile.tellg();
    }

    d_positions.endEdit();
    d_triangles.endEdit();
    d_normals.endEdit();

    dataFile.close();

    dmsg_info() << "done!" ;

    return true;
}




} // namespace loader

} // namespace component

} // namespace sofa


