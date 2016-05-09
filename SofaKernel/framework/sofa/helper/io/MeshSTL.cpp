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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/io/MeshSTL.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
using std::cout;
using std::endl;

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
        std::cerr << "File " << filename << " not found " << std::endl;
        return;
    }
    loaderType = "stl";
    std::ifstream file(filename.c_str());
    std::string token;
    if (file.good())
    {
        file >> token;
        if (token == "solid")
            readSTL(filename);
        else
            readBinarySTL(filename);

        // announce the model statistics
#ifndef NDEBUG
        std::cout << " Vertices: " << vertices.size() << std::endl;
        std::cout << " Normals: " << normals.size() << std::endl;
        std::cout << " Texcoords: " << texCoords.size() << std::endl;
        std::cout << " Triangles: " << facets.size() << std::endl;
#endif
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
            #ifndef NDEBUG
            std::cout << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">\n";
            #endif
        }
    }
    else
        std::cerr << "Error: MeshSTL: Cannot read file '" << filename << "'." << std::endl;

    file.close();
}

void MeshSTL::readSTL (const std::string &filename)
{
    /* http://www.ennex.com/~fabbers/StL.asp */
    #ifndef NDEBUG
    std::size_t namepos = filename.find_last_of("/");
    std::string name = filename.substr(namepos+1);
    cout << "Reading STL file : " << name << endl;
    #endif

    vector< vector<int> > vertNormTexIndices;
    vector<int> vIndices, nIndices, tIndices;
    Vec3d result;

    std::ifstream file(filename.c_str());
    std::string line;

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
            bool find = false;
            for (unsigned int i=0; i<vertices.size(); ++i)
                if ( (result[0] == vertices[i][0]) && (result[1] == vertices[i][1])  && (result[2] == vertices[i][2]))
                {
                    find = true;
                    vIndices.push_back(i);
                    break;
                }

            if (!find)
            {
                vertices.push_back(result);
                vIndices.push_back((int)vertices.size()-1);
            }

            // Useless but necessary to work -- need to be fixed properly
            tIndices.push_back(0);
            nIndices.push_back(0);
        }
        else if (token == "endfacet")
        {
            vertNormTexIndices.push_back (vIndices);
            vertNormTexIndices.push_back (nIndices);
            vertNormTexIndices.push_back (tIndices);
            facets.push_back(vertNormTexIndices);
            nIndices.clear();
            tIndices.clear();
            vIndices.clear();
            vertNormTexIndices.clear();
        }
        else if (token == "endsolid" || token == "end")
            break;

        else
        {
            // std::cerr << "readSTL : Unknown token for line " << line << std::endl;
        }
    }

}

void MeshSTL::readBinarySTL (const std::string &filename)
{
    /* Based on MeshSTLLoader */
    /* http://www.ennex.com/~fabbers/StL.asp */
    #ifndef NDEBUG
    std::size_t namepos = filename.find_last_of("/");
    std::string name = filename.substr(namepos+1);
    std::cout << "Reading binary STL file : " << name << std::endl;
    #endif

    std::ifstream dataFile(filename.c_str(), std::ios::in | std::ios::binary);
    std::streampos position = 0;
    std::streampos length;
    unsigned long int nbrFacet;
    vector< vector<int> > vertNormTexIndices;
    vector<int> vIndices, nIndices, tIndices;
    Vec3f result;

    // Get length of file
    dataFile.seekg(0, std::ios::end);
    length = dataFile.tellg();
    dataFile.seekg(0, std::ios::beg);

    // Skipping header file
    char buffer[256];
    dataFile.read(buffer, 80);

    // Get number of facets
    dataFile.read((char*)&nbrFacet, 4);

    // Parsing facets
    for (unsigned int i = 0; i<nbrFacet; ++i)
    {
        // Get normal
        dataFile.read((char*)&result[0], 4);
        dataFile.read((char*)&result[1], 4);
        dataFile.read((char*)&result[2], 4);
        //normals.push_back(result);

        // Get vertex
        for (unsigned int j = 0; j<3; ++j)
        {
            dataFile.read((char*)&result[0], 4);
            dataFile.read((char*)&result[1], 4);
            dataFile.read((char*)&result[2], 4);

            bool find = false;
            for (unsigned int k=0; k<vertices.size(); ++k)
                if ( (result[0] == vertices[k][0]) && (result[1] == vertices[k][1])  && (result[2] == vertices[k][2]))
                {
                    find = true;
                    vIndices.push_back(k);
                    break;
                }

            if (!find)
            {
                vertices.push_back(result);
                vIndices.push_back((int)vertices.size()-1);
            }

            // Useless but necessary to work -- need to be fixed properly
            tIndices.push_back(0);
            nIndices.push_back(0);
        }


        // Attribute byte count
        unsigned int count;
        dataFile.read((char*)&count, 2);

        vertNormTexIndices.push_back (vIndices);
        vertNormTexIndices.push_back (nIndices);
        vertNormTexIndices.push_back (tIndices);
        facets.push_back(vertNormTexIndices);
        vIndices.clear();
        nIndices.clear();
        tIndices.clear();
        vertNormTexIndices.clear();

        // Security -- End of file ?
        position = dataFile.tellg();
        if (position == length)
            break;
    }
}


} // namespace io

} // namespace helper

} // namespace sofa

