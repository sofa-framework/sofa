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
#include <sofa/helper/io/File.h>
#include <sofa/helper/io/MeshGmsh.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/helper/logging/Messaging.h>
#include <istream>
#include <fstream>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

SOFA_DECL_CLASS(MeshGmsh)

Creator<Mesh::FactoryMesh, MeshGmsh> MeshGmshClass("gmsh");

void MeshGmsh::init (std::string filename)
{
    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error("MeshGmsh") << "File " << filename << " not found.";
        return;
    }
    loaderType = "gmsh";

    std::cout << "passe bien la: MeshGmsh::init "<< std::endl;

    std::ifstream file(filename);
    if (!file.good()) return;

    int gmshFormat = 0;

    std::string cmd;
    file >> cmd;

    if (cmd == "$MeshFormat") // Reading gmsh 2.0 file
    {
        gmshFormat = 2;
        std::string line;
        std::getline(file, line); // we don't care about this line
        if (line == "") std::getline(file, line);
        file >> cmd;
        if (cmd != "$EndMeshFormat") // it should end with $EndMeshFormat
        {
            file.close();
            return;
        }
        else
        {
            file >> cmd;
        }
    }
    else
    {
        gmshFormat = 1;
    }

    readGmsh(file, gmshFormat);
    file.close();
}

bool MeshGmsh::readGmsh(std::ifstream &file, const unsigned int gmshFormat)
{
    int npoints = 0;
    int nlines = 0;
    int ntris = 0;
    int nquads = 0;
    int ntetrahedra = 0;
    int ncubes = 0;

    std::string cmd;

    file >> npoints; //nb points
    //setNbPoints(npoints); 

    std::vector<int> pmap;
    for (int i = 0; i<npoints; ++i)
    {
        int index = i;
        double x, y, z;
        file >> index >> x >> y >> z;
        m_vertices.push_back(sofa::defaulttype::Vector3(x, y, z));
        //addPoint(x, y, z);
        if ((int)pmap.size() <= index) pmap.resize(index + 1);
        pmap[index] = i;
    }

    file >> cmd;
    if (cmd != "$ENDNOD" && cmd != "$EndNodes")
    {
        msg_error("MeshGmsh") << "'$ENDNOD' or '$EndNodes' expected, found '" << cmd << "'";
        return false;
    }

    file >> cmd;
    if (cmd != "$ELM" && cmd != "$Elements")
    {
        msg_error("MeshGmsh") << "'$ELM' or '$Elements' expected, found '" << cmd << "'";
        return false;
    }



    int nelems = 0;
    file >> nelems;
    for (int i = 0; i<nelems; ++i)
    {
        int index = -1, etype = -1, nnodes = -1, ntags = -1, tag = -1;
        if (gmshFormat == 1)
        {
            // version 1.0 format is
            // elm-number elm-type reg-phys reg-elem number-of-nodes <node-number-list ...>
            int rphys = -1, relem = -1;
            file >> index >> etype >> rphys >> relem >> nnodes;
        }
        else if (gmshFormat == 2)
        {
            // version 2.0 format is
            // elm-number elm-type number-of-tags < tag > ... node-number-list
            file >> index >> etype >> ntags;

            for (int t = 0; t<ntags; t++)
            {
                file >> tag;
                // read the tag but don't use it
            }

            switch (etype)
            {
            case 1: // Line
                nnodes = 2;
                break;
            case 2: // Triangle
                nnodes = 3;
                break;
            case 3: // Quad
                nnodes = 4;
                break;
            case 4: // Tetra
                nnodes = 4;
                break;
            case 5: // Hexa
                nnodes = 8;
                break;
            case 15: // Point
                nnodes = 1;
                break;
            default:
                msg_error("MeshGmsh") << "Elements of type 1, 2, 3, 4, 5, or 6 expected. Element of type " << etype << " found.";
                nnodes = 0;
            }
        }

        helper::vector<int> nodes;
        nodes.resize(nnodes);
        for (int n = 0; n<nnodes; ++n)
        {
            int t = 0;
            file >> t;
            nodes[n] = (((unsigned int)t)<pmap.size()) ? pmap[t] : 0;
        }
        switch (etype)
        {
        case 1: // Line
            //addLine(nodes[0], nodes[1]);
            m_edges.push_back(Topology::Edge(nodes[0], nodes[1]));
            ++nlines;
            break;
        case 2: // Triangle
            //addTriangle(nodes[0], nodes[1], nodes[2]);
            m_triangles.push_back(Topology::Triangle(nodes[0], nodes[1], nodes[2]));
            ++ntris;
            break;
        case 3: // Quad
            //addQuad(nodes[0], nodes[1], nodes[2], nodes[3]);
            m_quads.push_back(Topology::Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
            ++nquads;
            break;
        case 4: // Tetra
            //addTetra(nodes[0], nodes[1], nodes[2], nodes[3]);
            m_tetrahedra.push_back(Topology::Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
            ++ntetrahedra;
            break;
        case 5: // Hexa
            //addCube(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]);
            m_hexahedra.push_back(Topology::Hexahedron(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]));
            ++ncubes;
            break;
        default:
            //if the type is not handled, skip rest of the line
            std::string tmp;
            std::getline(file, tmp);
        }
    }

    file >> cmd;
    if (cmd != "$ENDELM" && cmd != "$EndElements")
    {
        msg_error("MeshGmsh") << "'$ENDELM' or '$EndElements' expected, found '" << cmd << "'";
        return false;
    }

    return true;
}

} // namespace io

} // namespace helper

} // namespace sofa

