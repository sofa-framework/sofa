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
#include <sofa/helper/io/File.h>
#include <sofa/helper/io/MeshGmsh.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/helper/logging/Messaging.h>
#include <istream>
#include <fstream>
#include <string>
#include <sofa/helper/narrow_cast.h>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::type;
using namespace sofa::topology;

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
    std::getline(file, cmd); //First line should be the start of the $MeshFormat section
    if (cmd.length() >= 11 && cmd.substr(0, 11) == "$MeshFormat") // Reading gmsh
    {
        // NB: .msh file header line for version >= 2 can be "$MeshFormat", "$MeshFormat\r", "$MeshFormat \r"
        std::string version;
        std::getline(file, version); // Getting the version line (e.g. 4.1 0 8)
        gmshFormat = std::stoul(version.substr( 0, version.find(" ")) ); // Retrieving the mesh format, keeping only the integer part
        std::getline(file, cmd);

        if (cmd.length() < 14 || cmd.substr(0, 14) != std::string("$EndMeshFormat")) // it should end with "$EndMeshFormat" or "$EndMeshFormat\r"
        {
            msg_error("MeshGmsh") << "No $EndMeshFormat flag found at the end of the file. Closing File";
            file.close();
            return;
        }
        else
        {
            // Reading the file until the node section is hit. In recent versions of MSH file format,
            // we may encounter various sections between $MeshFormat and $Nodes
            while (cmd.length() < 6 || cmd.substr(0, 6) != std::string("$Nodes")) // can be "$Nodes" or "$Nodes\r"
            {
                std::getline(file, cmd); // First Command
                if (file.eof())
                {
                    msg_error("MeshGmsh") << "End of file reached without finding the $Nodes section expected in MSH file format. Closing file.";
                    file.close();
                    return;
                }
            }
        }
    }
    else
    {
        // Legacy MSh format version 1 directly starts with the Nodes section
        // https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-1-_0028Legacy_0029
        gmshFormat = 1;
        // The next line is already the first line of the $Nodes section. The file can be passed
        // to readGmdh in its current state
    }

    readGmsh(file, gmshFormat);
    file.close();
}


void MeshGmsh::addInGroup(type::vector< sofa::type::PrimitiveGroup>& group, int tag, std::size_t /*eid*/)
{
    for (std::size_t i = 0; i<group.size(); i++) {
        if (tag == group[i].p0) {
            group[i].nbp++;
            return;
        }
    }

    std::stringstream ss;
    const std::string s;
    ss << tag;

    group.push_back(sofa::type::PrimitiveGroup(tag, 1, s, s, -1));
}

void MeshGmsh::normalizeGroup(type::vector< sofa::type::PrimitiveGroup>& group)
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

    std::string cmd;

    if (gmshFormat <= 2)
    {
        // --- Loading Vertices ---
        file >> npoints; //nb points

        std::vector<int> pmap; // map for reordering vertices possibly not well sorted
        for (int i = 0; i < npoints; ++i)
        {
            int index = i;
            double x, y, z;
            file >> index >> x >> y >> z;
            m_vertices.push_back(sofa::type::Vec3(x, y, z));
            if ((int)pmap.size() <= index) pmap.resize(index + 1);
            pmap[index] = i; // In case of hole or swit
        }

        file >> cmd;
        if (cmd.length() < 7 || cmd.substr(0, 7) != "$ENDNOD") // can be "$ENDNOD" or "$ENDNOD\r"
        {
            if (cmd.length() < 9 || cmd.substr(0, 9) != "$EndNodes") // can be "$EndNodes" or "$EndNodes\r"
            {
                msg_error("MeshGmsh") << "'$ENDNOD' or '$EndNodes' expected, found '" << cmd << "'";
                return false;
            }
        }

        // --- Loading Elements ---
        file >> cmd;
        if (cmd.length() < 4 || cmd.substr(0, 4) != "$ELM") // can be "$ELM" or "$ELM\r"
        {
            if (cmd.length() < 9 || cmd.substr(0, 9) != "$Elements") // can be "$ELM" or "$ELM\r"
            {
                msg_error("MeshGmsh") << "'$ELM' or '$Elements' expected, found '" << cmd << "'";
                return false;
            }
        }

        int nelems = 0;
        file >> nelems;

        for (int i = 0; i < nelems; ++i) // for each elem
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

                for (int t = 0; t < ntags; t++)
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

            type::vector<unsigned int> nodes;
            nodes.resize(nnodes);
            constexpr unsigned int edgesInQuadraticTriangle[3][2] = { { 0,1 },{ 1,2 },{ 2,0 } };
            constexpr unsigned int edgesInQuadraticTetrahedron[6][2] = { { 0,1 },{ 1,2 },{ 0,2 },{ 0,3 },{ 2,3 },{ 1,3 } };
            std::set<Edge> edgeSet;
            size_t j;
            for (int n = 0; n < nnodes; ++n)
            {
                int t = 0;
                file >> t;
                nodes[n] = (((unsigned int)t) < pmap.size()) ? pmap[t] : 0;
            }

            switch (etype)
            {
            case 1: // Line
                addInGroup(m_edgesGroups, tag, m_edges.size());
                m_edges.push_back(Edge(nodes[0], nodes[1]));
                break;
            case 2: // Triangle
                addInGroup(m_trianglesGroups, tag, m_triangles.size());
                m_triangles.push_back(Triangle(nodes[0], nodes[1], nodes[2]));
                break;
            case 3: // Quad
                addInGroup(m_quadsGroups, tag, m_quads.size());
                m_quads.push_back(Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
                break;
            case 4: // Tetra
                addInGroup(m_tetrahedraGroups, tag, m_tetrahedra.size());
                m_tetrahedra.push_back(Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                break;
            case 5: // Hexa
                addInGroup(m_hexahedraGroups, tag, m_hexahedra.size());
                m_hexahedra.push_back(Hexahedron(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]));
                break;
            case 8: // quadratic edge
                addInGroup(m_edgesGroups, tag, m_edges.size());
                m_edges.push_back(Edge(nodes[0], nodes[1]));
                {
                    HighOrderEdgePosition hoep;
                    hoep[0] = nodes[2];
                    hoep[1] = sofa::helper::narrow_cast<PointID>(m_edges.size() - 1);
                    hoep[2] = 1;
                    hoep[3] = 1;
                    m_highOrderEdgePositions.push_back(hoep);
                }
                break;
            case 9: // quadratic triangle
                addInGroup(m_trianglesGroups, tag, m_triangles.size());
                m_triangles.push_back(Triangle(nodes[0], nodes[1], nodes[2]));
                {
                    HighOrderEdgePosition hoep;
                    for (j = 0; j < 3; ++j) {
                        auto v0 = std::min(nodes[edgesInQuadraticTriangle[j][0]],
                            nodes[edgesInQuadraticTriangle[j][1]]);
                        auto v1 = std::max(nodes[edgesInQuadraticTriangle[j][0]],
                            nodes[edgesInQuadraticTriangle[j][1]]);
                        Edge e(v0, v1);
                        if (edgeSet.find(e) == edgeSet.end()) {
                            edgeSet.insert(e);
                            m_edges.push_back(Edge(v0, v1));
                            hoep[0] = nodes[j + 3];
                            hoep[1] = sofa::helper::narrow_cast<PointID>(m_edges.size() - 1);
                            hoep[2] = 1;
                            hoep[3] = 1;
                            m_highOrderEdgePositions.push_back(hoep);
                        }
                    }
                }
                break;
            case 11: // quadratic tetrahedron
                addInGroup(m_tetrahedraGroups, tag, m_tetrahedra.size());
                m_tetrahedra.push_back(Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                {
                    HighOrderEdgePosition hoep;
                    for (j = 0; j < 6; ++j) {
                        auto v0 = std::min(nodes[edgesInQuadraticTetrahedron[j][0]],
                            nodes[edgesInQuadraticTetrahedron[j][1]]);
                        auto v1 = std::max(nodes[edgesInQuadraticTetrahedron[j][0]],
                            nodes[edgesInQuadraticTetrahedron[j][1]]);
                        Edge e(v0, v1);
                        if (edgeSet.find(e) == edgeSet.end()) {
                            edgeSet.insert(e);
                            m_edges.push_back(Edge(v0, v1));
                            hoep[0] = nodes[j + 4];
                            hoep[1] = sofa::helper::narrow_cast<PointID>(m_edges.size() - 1);
                            hoep[2] = 1;
                            hoep[3] = 1;
                            m_highOrderEdgePositions.push_back(hoep);
                        }
                    }
                }
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
    }

    else // gmshFormat >= 4
    {
        // --- Parsing the $Nodes section --- //

        std::getline(file, cmd); // Getting first line of $Nodes
        std::istringstream nodesHeader(cmd);
        unsigned int nbEntityBlocks, nbNodes, minNodeTag, maxNodeTag;
        nodesHeader >> nbEntityBlocks >> nbNodes >> minNodeTag >> maxNodeTag;

        for (unsigned int entityIndex = 0; entityIndex < nbEntityBlocks; entityIndex++) // looping over the entity blocks
        {
            std::getline(file, cmd); // Reading the entity line
            std::istringstream entitySummary(cmd);
            unsigned int entityDim, entityTag, parametric, nbNodesInBlock;
            entitySummary >> entityDim >> entityTag >> parametric >> nbNodesInBlock;

            for (unsigned int nodeIndex = 0; nodeIndex < nbNodesInBlock; nodeIndex++)
                std::getline(file, cmd); // Reading the node indices lines
            for (unsigned int nodeIndex = 0; nodeIndex < nbNodesInBlock; nodeIndex++)
            {
                std::getline(file, cmd); // Reading the node coordinates
                std::istringstream coordinates(cmd);
                double x, y, z;
                coordinates >> x >> y >> z;
                m_vertices.push_back(sofa::type::Vec3(x, y, z));
            }
        }

        std::getline(file, cmd);
        if (cmd.substr(0, 9) != "$EndNodes")
        {
            msg_error("MeshGmsh") << "'$EndNodes' expected, found '" << cmd << "'";
            return false;
        }


        // --- Parsing the $Elements section --- //

        std::getline(file, cmd);
        if (cmd.substr(0, 9) != "$Elements")
        {
            msg_error("MeshGmsh") << "'$Elements' expected, found '" << cmd << "'";
            return false;
        }

        std::getline(file, cmd); // Getting first line of $Elements
        std::istringstream elementsHeader(cmd);
        unsigned int nbElements, minElementTag, maxElementTag;
        elementsHeader >> nbEntityBlocks >> nbElements >> minElementTag >> maxElementTag;

        // Common information to add second order triangles (elementType = 9) and tetrahedra (elementType = 11)
        const unsigned int edgesInQuadraticTriangle[3][2] = { { 0,1 },{ 1,2 },{ 2,0 } };
        const unsigned int edgesInQuadraticTetrahedron[6][2] = { { 0,1 },{ 1,2 },{ 0,2 },{ 0,3 },{ 2,3 },{ 1,3 } };
        std::set<Edge> edgeSet;

        for (unsigned int entityIndex = 0; entityIndex < nbEntityBlocks; entityIndex++) // looping over the entity blocks
        {
            std::getline(file, cmd); // Reading the entity line
            std::istringstream entitySummary(cmd);
            unsigned int entityDim, entityTag, nbElementsInBlock, elementType;
            entitySummary >> entityDim >> entityTag >> elementType >> nbElementsInBlock;

            unsigned int nnodes = 0;
            switch (elementType)
            {
            case 1: // Line
                nnodes = 2;
                break;
            case 2: // Triangle
                nnodes = 3;
                break;
            case 3: // Quadrangle
                nnodes = 4;
                break;
            case 4: // Tetrahedron
                nnodes = 4;
                break;
            case 5: // Hexahedron
                nnodes = 8;
                break;
            case 6: // Prism
                nnodes = 6;
                break;
            case 8: // Second order line
                nnodes = 3;
                break;
            case 9: // Second order triangle
                nnodes = 6;
                break;
            case 11: // Second order tetrahedron
                nnodes = 10;
                break;
            case 15: // Point
                nnodes = 1;
                break;
            default:
                msg_error("MeshGmsh") << "Elements of type 1, 2, 3, 4, 5, 8, 9, 11 or 15 expected. Element of type " << elementType << " found.";
                // nnodes = 0;
            }

            for (unsigned int elemIndex = 0; elemIndex < nbElementsInBlock; elemIndex++)
            {
                std::getline(file, cmd); // Reading the element info
                std::istringstream elementInfo(cmd);
                unsigned int elementTag;
                elementInfo >> elementTag;

                type::vector<unsigned int> nodes;
                unsigned int nodeId = 0;
                nodes.resize(nnodes);

                for (unsigned int i = 0; i < nnodes; i++)
                {
                    elementInfo >> nodeId;
                    nodes[i] = nodeId-1; //To account for the fact that node indices in the MSH file format start with 1 instead of 0
                }

                switch (elementType)
                {
                case 1: // Line
                    addInGroup(m_edgesGroups, elementTag, m_edges.size());
                    m_edges.push_back(Edge(nodes[0], nodes[1]));
                    break;
                case 2: // Triangle
                    addInGroup(m_trianglesGroups, elementTag, m_triangles.size());
                    m_triangles.push_back(Triangle(nodes[0], nodes[1], nodes[2]));
                    break;
                case 3: // Quadrangle
                    addInGroup(m_quadsGroups, elementTag, m_quads.size());
                    m_quads.push_back(Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
                    break;
                case 4: // Tetrahedron
                    addInGroup(m_tetrahedraGroups, elementTag, m_tetrahedra.size());
                    m_tetrahedra.push_back(Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                    break;
                case 5: // Hexahedron
                    addInGroup(m_hexahedraGroups, elementTag, m_hexahedra.size());
                    m_hexahedra.push_back(Hexahedron(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]));
                    break;
                case 8: // Second order line
                    addInGroup(m_edgesGroups, elementTag, m_edges.size());
                    m_edges.push_back(Edge(nodes[0], nodes[1]));
                    {
                        HighOrderEdgePosition hoep;
                        hoep[0] = nodes[2];
                        hoep[1] = sofa::helper::narrow_cast<PointID>(m_edges.size() - 1);
                        hoep[2] = 1;
                        hoep[3] = 1;
                        m_highOrderEdgePositions.push_back(hoep);
                    }
                    break;
                case 9: // Second order triangle
                    addInGroup(m_trianglesGroups, elementTag, m_triangles.size());
                    m_triangles.push_back(Triangle(nodes[0], nodes[1], nodes[2]));
                    {
                        HighOrderEdgePosition hoep;
                        for (size_t j = 0; j < 3; ++j)
                        {
                            auto v0 = std::min(nodes[edgesInQuadraticTriangle[j][0]],
                                nodes[edgesInQuadraticTriangle[j][1]]);
                            auto v1 = std::max(nodes[edgesInQuadraticTriangle[j][0]],
                                nodes[edgesInQuadraticTriangle[j][1]]);
                            Edge e(v0, v1);
                            if (edgeSet.find(e) == edgeSet.end())
                            {
                                edgeSet.insert(e);
                                m_edges.push_back(Edge(v0, v1));
                                hoep[0] = nodes[j + 3];
                                hoep[1] = sofa::helper::narrow_cast<PointID>(m_edges.size() - 1);
                                hoep[2] = 1;
                                hoep[3] = 1;
                                m_highOrderEdgePositions.push_back(hoep);
                            }
                        }
                    }
                    break;
                case 11: // Second order tetrahedron
                    addInGroup(m_tetrahedraGroups, elementTag, m_tetrahedra.size());
                    m_tetrahedra.push_back(Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                    {
                        HighOrderEdgePosition hoep;
                        for (size_t j = 0; j < 6; ++j)
                        {
                            auto v0 = std::min(nodes[edgesInQuadraticTetrahedron[j][0]],
                                nodes[edgesInQuadraticTetrahedron[j][1]]);
                            auto v1 = std::max(nodes[edgesInQuadraticTetrahedron[j][0]],
                                nodes[edgesInQuadraticTetrahedron[j][1]]);
                            Edge e(v0, v1);
                            if (edgeSet.find(e) == edgeSet.end())
                            {
                                edgeSet.insert(e);
                                m_edges.push_back(Edge(v0, v1));
                                hoep[0] = nodes[j + 4];
                                hoep[1] = sofa::helper::narrow_cast<PointID>(m_edges.size() - 1);
                                hoep[2] = 1;
                                hoep[3] = 1;
                                m_highOrderEdgePositions.push_back(hoep);
                            }
                        }
                    }
                    break;
                // default: if the type is not handled, nothing to be done
                }
            } // end of loop over the elements in one entity block
        } //end of loop over the entity blocks

        normalizeGroup(m_edgesGroups);
        normalizeGroup(m_trianglesGroups);
        normalizeGroup(m_tetrahedraGroups);
        normalizeGroup(m_hexahedraGroups);
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
