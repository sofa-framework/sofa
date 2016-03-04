/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/ObjectFactory.h>
#include <SofaLoader/MeshSTLLoader.h>
#include <sofa/core/visual/VisualParams.h>

#include <iostream>
//#include <fstream> // we can't use iostream because the windows implementation gets confused by the mix of text and binary
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
    , _headerSize(initData(&_headerSize, (unsigned int)80, "headerSize","Size of the header binary file (just before the number of facet)."))
    , _forceBinary(initData(&_forceBinary, (bool)false, "forceBinary","Force reading in binary mode. Even in first keyword of the file is solid."))
{
}



bool MeshSTLLoader::load()
{
    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();
    std::ifstream file(filename);


    if (!file.good())
    {
        serr << "Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    std::string test;
    file >> test;

    if (test == "solid" && !_forceBinary.getValue())
        fileRead = this->readSTL(filename);
    else
        fileRead = this->readBinarySTL(filename); // -- Reading binary file


    file.close();
    return fileRead;
}


bool MeshSTLLoader::readBinarySTL(const char *filename)
{
    std::cout << "reading binary STL file" << std::endl;
    std::ifstream dataFile (filename, std::ios::in | std::ios::binary);

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& my_normals = *(normals.beginEdit());
    helper::vector<Triangle >& my_triangles = *(triangles.beginEdit());

    // get length of file
    dataFile.seekg(0, std::ios::end);
    std::streampos length = dataFile.tellg();
    dataFile.seekg(0, std::ios::beg);


    // Skipping header file
    char buffer[256];
    dataFile.read(buffer, _headerSize.getValue());
    sout << "Header binary file: "<< buffer << sendl;

    uint32_t nbrFacet;
    dataFile.read((char*)&nbrFacet, 4);

    std::streampos position = 0;
    // Parsing facets
    std::cout << "Reading file...";
    for (uint32_t i = 0; i<nbrFacet; ++i)
    {
        Triangle the_tri;
        sofa::defaulttype::Vec3f vertex, normals;

        // Normal:
        dataFile.read((char*)&normals[0], 4);
        dataFile.read((char*)&normals[1], 4);
        dataFile.read((char*)&normals[2], 4);
        my_normals.push_back(normals);

        // Vertices:
        for (size_t j = 0; j<3; ++j)
        {
            dataFile.read((char*)&vertex[0], 4);
            dataFile.read((char*)&vertex[1], 4);
            dataFile.read((char*)&vertex[2], 4);

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

        // Triangle:
        my_triangles.push_back(the_tri);

        // Atribute byte count
        uint16_t count;
        dataFile.read((char*)&count, 2);

        // Security:
        position = dataFile.tellg();
        if (position == length)
            break;
    }
    std::cout << "done!" << std::endl;

    positions.endEdit();
    triangles.endEdit();
    normals.endEdit();

    return true;
}


bool MeshSTLLoader::readSTL(const char *filename)
{
    sout << "reading STL file" << sendl;

    // Init
    std::ifstream dataFile (filename);
    std::string buffer;
    std::string name; // name of the solid, needed?

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& my_normals = *(normals.beginEdit());
    helper::vector<Triangle >& my_triangles = *(triangles.beginEdit());


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

    positions.endEdit();
    triangles.endEdit();
    normals.endEdit();

    return true;
}




} // namespace loader

} // namespace component

} // namespace sofa

