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
#include <string>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;


Creator<Mesh::FactoryMesh, MeshGmsh> MeshGmshClass("gmsh");

void MeshGmsh::init (std::string filename)
{
    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error("MeshGmsh") << "File " << filename << " not found.";
        return;
    }
    loaderType = "gmsh";

    std::ifstream file(filename);
    if (!file.good()) return;

    unsigned int gmshFormat = 0;
    std::string cmd;

    // -- Looking for Gmsh version of this file.
    std::getline(file, cmd); //Version
    std::istringstream versionReader(cmd);
    std::string version;
    versionReader >> version;
    if (version == "$MeshFormat") // Reading gmsh 2.0 file
    {
        gmshFormat = 2;
        std::string line;
        std::getline(file, line); // we don't nedd this line
        std::getline(file, cmd);
        std::istringstream endMeshReader(cmd);
        std::string endMesh;
        endMeshReader >> endMesh;

        if (endMesh != std::string("$EndMeshFormat")) // it should end with $EndMeshFormat
        {
            file.close();
            return;
        }
        else
        {
            std::getline(file, cmd); // First Command
        }
    }
    else
    {
        gmshFormat = 1;
    }

    readGmsh(file, gmshFormat);
    file.close();
}


void MeshGmsh::addInGroup(helper::vector< sofa::core::loader::PrimitiveGroup>& group, int tag, int /*eid*/) 
{
    for (unsigned i = 0; i<group.size(); i++) {
        if (tag == group[i].p0) {
            group[i].nbp++;
            return;
        }
    }

    std::stringstream ss;
    std::string s;
    ss << tag;

    group.push_back(sofa::core::loader::PrimitiveGroup(tag, 1, s, s, -1));
}

void MeshGmsh::normalizeGroup(helper::vector< sofa::core::loader::PrimitiveGroup>& group) 
{
    int start = 0;
    for (unsigned i = 0; i<group.size(); i++) {
        group[i].p0 = start;
        start += group[i].nbp;
    }
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

    // --- Loading Vertices ---
    file >> npoints; //nb points

    std::vector<int> pmap; // map for reordering vertices possibly not well sorted
    for (int i = 0; i<npoints; ++i)
    {
        int index = i;
        double x, y, z;
        file >> index >> x >> y >> z;
        m_vertices.push_back(sofa::defaulttype::Vector3(x, y, z));
        if ((int)pmap.size() <= index) pmap.resize(index + 1);
        pmap[index] = i; // In case of hole or swit
    }
    
    file >> cmd;
    if (cmd != "$ENDNOD" && cmd != "$EndNodes")
    {
        msg_error("MeshGmsh") << "'$ENDNOD' or '$EndNodes' expected, found '" << cmd << "'";
        return false;
    }

    // --- Loading Elements ---
    file >> cmd;
    if (cmd != "$ELM" && cmd != "$Elements")
    {
        msg_error("MeshGmsh") << "'$ELM' or '$Elements' expected, found '" << cmd << "'";
        return false;
    }

    int nelems = 0;
    file >> nelems;

    for (int i = 0; i<nelems; ++i) // for each elem
    {
        int index = -1, etype = -1, nnodes = -1, ntags = -1, tag = -1;
        if (gmshFormat == 1)
        {
            // version 1.0 format is
            // elm-number elm-type reg-phys reg-elem number-of-nodes <node-number-list ...>
            int rphys = -1, relem = -1;
            file >> index >> etype >> rphys >> relem >> nnodes;
        }
        else /*if (gmshFormat == 2)*/
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
            case 15: //point
                nnodes = 1;
                break;
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
            case 8: // Quadratic edge
                nnodes = 3;
                break;
            case 9: // Quadratic Triangle
                nnodes = 6;
                break;
            case 11: // Quadratic Tetrahedron
                nnodes = 10;
                break;
            default:
                msg_error("MeshGmsh") << "Elements of type 1, 2, 3, 4, 5, or 6 expected. Element of type " << etype << " found.";
                nnodes = 0;
            }
        }

        helper::vector<unsigned int> nodes;
        nodes.resize(nnodes);
        const unsigned int edgesInQuadraticTriangle[3][2] = { { 0,1 },{ 1,2 },{ 2,0 } };
        const unsigned int edgesInQuadraticTetrahedron[6][2] = { { 0,1 },{ 1,2 },{ 0,2 },{ 0,3 },{ 2,3 },{ 1,3 } };
        std::set<Topology::Edge> edgeSet;
        size_t j;
        for (int n = 0; n<nnodes; ++n)
        {
            int t = 0;
            file >> t;
            nodes[n] = (((unsigned int)t)<pmap.size()) ? pmap[t] : 0;
        }

        switch (etype)
        {
        case 1: // Line
            addInGroup(m_edgesGroups, tag, m_edges.size());
            m_edges.push_back(Topology::Edge(nodes[0], nodes[1]));
            ++nlines;
            break;
        case 2: // Triangle
            addInGroup(m_trianglesGroups, tag, m_triangles.size());
            m_triangles.push_back(Topology::Triangle(nodes[0], nodes[1], nodes[2]));
            ++ntris;
            break;
        case 3: // Quad
            addInGroup(m_quadsGroups, tag, m_quads.size());
            m_quads.push_back(Topology::Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
            ++nquads;
            break;
        case 4: // Tetra
            addInGroup(m_tetrahedraGroups, tag, m_tetrahedra.size());
            m_tetrahedra.push_back(Topology::Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
            ++ntetrahedra;
            break;
        case 5: // Hexa
            addInGroup(m_hexahedraGroups, tag, m_hexahedra.size());
            m_hexahedra.push_back(Topology::Hexahedron(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]));
            ++ncubes;
            break;
        case 8: // quadratic edge
            addInGroup(m_edgesGroups, tag, m_edges.size());
            m_edges.push_back(Topology::Edge(nodes[0], nodes[1]));
            {
                HighOrderEdgePosition hoep;
                hoep[0] = nodes[2];
                hoep[1] = m_edges.size() - 1;
                hoep[2] = 1;
                hoep[3] = 1;
                m_highOrderEdgePositions.push_back(hoep);
            }
            ++nlines;
            break;
        case 9: // quadratic triangle
            addInGroup(m_trianglesGroups, tag, m_triangles.size());
            m_triangles.push_back(Topology::Triangle(nodes[0], nodes[1], nodes[2]));
            {
                HighOrderEdgePosition hoep;
                for (j = 0; j<3; ++j) {
                    size_t v0 = std::min(nodes[edgesInQuadraticTriangle[j][0]],
                        nodes[edgesInQuadraticTriangle[j][1]]);
                    size_t v1 = std::max(nodes[edgesInQuadraticTriangle[j][0]],
                        nodes[edgesInQuadraticTriangle[j][1]]);
                    Topology::Edge e(v0, v1);
                    if (edgeSet.find(e) == edgeSet.end()) {
                        edgeSet.insert(e);
                        m_edges.push_back(Topology::Edge(v0, v1));
                        hoep[0] = nodes[j + 3];
                        hoep[1] = m_edges.size() - 1;
                        hoep[2] = 1;
                        hoep[3] = 1;
                        m_highOrderEdgePositions.push_back(hoep);
                    }
                }
            }
            ++ntris;
            break;
        case 11: // quadratic tetrahedron
            addInGroup(m_tetrahedraGroups, tag, m_tetrahedra.size());
            m_tetrahedra.push_back(Topology::Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
            {
                HighOrderEdgePosition hoep;
                for (j = 0; j<6; ++j) {
                    size_t v0 = std::min(nodes[edgesInQuadraticTetrahedron[j][0]],
                        nodes[edgesInQuadraticTetrahedron[j][1]]);
                    size_t v1 = std::max(nodes[edgesInQuadraticTetrahedron[j][0]],
                        nodes[edgesInQuadraticTetrahedron[j][1]]);
                    Topology::Edge e(v0, v1);
                    if (edgeSet.find(e) == edgeSet.end()) {
                        edgeSet.insert(e);
                        m_edges.push_back(Topology::Edge(v0, v1));
                        hoep[0] = nodes[j + 4];
                        hoep[1] = m_edges.size() - 1;
                        hoep[2] = 1;
                        hoep[3] = 1;
                        m_highOrderEdgePositions.push_back(hoep);
                    }
                }
            }
            ++ntetrahedra;
            break;
        default:
            //if the type is not handled, skip rest of the line
            std::string tmp;
            std::getline(file, tmp);
        }
    }

    normalizeGroup(m_edgesGroups);
    normalizeGroup(m_trianglesGroups);
    normalizeGroup(m_tetrahedraGroups);
    normalizeGroup(m_hexahedraGroups);

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

