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
#include <sofa/component/io/mesh/MeshOffLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <fstream>

namespace sofa::component::io::mesh
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::helper;

void registerMeshOffLoader(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Specific mesh loader for Off file format.")
        .add< MeshOffLoader >());
}

bool MeshOffLoader::doLoad()
{
    msg_info() << "Loading OFF file: " << d_filename;

    bool fileRead = false;

    // -- Loading file
    const char* filename = d_filename.getFullPath().c_str();
    std::string cmd;
    std::ifstream file(filename);

    if (!file.good())
    {
        msg_error() << "Cannot read file '" << d_filename << "'.";
        return false;
    }

    file >> cmd;
    if (cmd != "OFF")
    {
        msg_error() << "Not a OFF file (header problem) '" << d_filename << "'.";
        return false;
    }

    // -- Reading file
    fileRead = this->readOFF (file,filename);
    file.close();

    return fileRead;
}

void MeshOffLoader::doClearBuffers() {}

bool MeshOffLoader::readOFF (std::ifstream &file, const char* /* filename */ )
{
    msg_info() << "MeshOffLoader::readOFF" ;

    auto my_positions = getWriteOnlyAccessor(d_positions);
    auto my_triangles = getWriteOnlyAccessor(d_triangles);
    auto my_quads = getWriteOnlyAccessor(d_quads);

    size_t numberOfVertices = 0, numberOfFaces = 0, numberOfEdges = 0;
    size_t currentNumberOfVertices = 0, currentNumberOfFaces = 0;
    Vec3d vertex;
    Triangle triangle;
    Quad quad;
    std::string line;

    while( !file.eof() && (numberOfVertices == 0) )
    {
        std::getline(file,line);

        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream values(line);
        values >> numberOfVertices >> numberOfFaces >> numberOfEdges;
    }

    msg_info() << "vertices = "<< numberOfVertices
               << "faces = "<< numberOfFaces
               << "edges = "<< numberOfEdges ;

    currentNumberOfVertices = 0;

    //Vertices
    while( !file.eof() &&  currentNumberOfVertices < numberOfVertices)
    {
        std::getline(file,line);

        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream values(line);

        values >> vertex[0] >> vertex[1] >> vertex[2];
        my_positions.push_back(Vec3(vertex[0],vertex[1], vertex[2]));

        currentNumberOfVertices++;
    }
    currentNumberOfFaces = 0;
    //Faces
    while( !file.eof() && currentNumberOfFaces < numberOfFaces)
    {
        std::getline(file,line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream values(line);
        size_t numberOfVerticesPerFace = 0;

        values >> numberOfVerticesPerFace;
        if (numberOfVerticesPerFace < 3 || numberOfVerticesPerFace > 4)
            continue;

        if (numberOfVerticesPerFace == 3)
        {
            values >> triangle[0] >> triangle[1] >> triangle[2];
            addTriangle(my_triangles.wref(), triangle);
        }
        if (numberOfVerticesPerFace == 4)
        {
            values >> quad[0] >> quad[1] >> quad[2] >> quad[3];
            addQuad(my_quads.wref(), quad);
        }
        currentNumberOfFaces++;
    }

    return true;
}
} //namespace sofa::component::io::mesh
