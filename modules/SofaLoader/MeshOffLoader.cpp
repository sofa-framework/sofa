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
#include <SofaLoader/MeshOffLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/SetDirectory.h>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshOffLoader)

int MeshOffLoaderClass = core::RegisterObject("Specific mesh loader for Off file format.")
        .add< MeshOffLoader >()
        ;

bool MeshOffLoader::load()
{
    sout << "Loading OFF file: " << m_filename << sendl;

    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();
    std::string cmd;
    std::ifstream file(filename);

    if (!file.good())
    {
        serr << "Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    file >> cmd;
    if (cmd != "OFF")
    {
        serr << "Not a OFF file (header problem) '" << m_filename << "'." << sendl;
        return false;
    }

    // -- Reading file
    fileRead = this->readOFF (file,filename);
    file.close();

    return fileRead;
}



bool MeshOffLoader::readOFF (std::ifstream &file, const char* /* filename */ )
{
    if(f_printLog.getValue())
        sout << "MeshOffLoader::readOFF" << sendl;

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());

    helper::vector<Triangle>& my_triangles = *(triangles.beginEdit());
    helper::vector<Quad>& my_quads = *(quads.beginEdit());

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

    if(f_printLog.getValue())
    {
        sout << "vertices = "<< numberOfVertices << sendl;
        sout << "faces = "<< numberOfFaces << sendl;
        sout << "edges = "<< numberOfEdges << sendl;
    }


    currentNumberOfVertices = 0;

    //Vertices
    while( !file.eof() &&  currentNumberOfVertices < numberOfVertices)
    {
        std::getline(file,line);

        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream values(line);

        values >> vertex[0] >> vertex[1] >> vertex[2];
        my_positions.push_back(Vector3(vertex[0],vertex[1], vertex[2]));

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
            addTriangle(&my_triangles, triangle);
        }
        if (numberOfVerticesPerFace == 4)
        {
            values >> quad[0] >> quad[1] >> quad[2] >> quad[3];
            addQuad(&my_quads, quad);
        }
        currentNumberOfFaces++;
    }

    positions.endEdit();
    triangles.endEdit();
    quads.endEdit();

    return true;
}
} // namespace loader

} // namespace component

} // namespace sofa

