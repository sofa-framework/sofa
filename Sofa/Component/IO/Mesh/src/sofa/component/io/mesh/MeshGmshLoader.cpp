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
#include <sofa/component/io/mesh/config.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/io/mesh/MeshGmshLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <fstream>
#include <sofa/helper/io/Mesh.h>


namespace sofa::component::io::mesh
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::helper;
using std::string;
using std::stringstream;

int MeshGmshLoaderClass = core::RegisterObject("Specific mesh loader for Gmsh file format.")
        .add< MeshGmshLoader >()
        ;

bool MeshGmshLoader::doLoad()
{
    string cmd;
    unsigned int gmshFormat = 0;

    if (!canLoad())
    {
        msg_error(this) << "Can't load file " << d_filename.getFullPath().c_str();
        return false;
    }
    // -- Loading file
    const char* filename = d_filename.getFullPath().c_str();
    std::ifstream file(filename);

    // -- Looking for Gmsh version of this file.
    std::getline(file, cmd);
    if (cmd.length() >= 11 && cmd.substr(0, 11) == "$MeshFormat") // Reading gmsh
    {
        // NB: .msh file header line for version >= 2 can be "$MeshFormat", "$MeshFormat\r", "$MeshFormat \r"
        string version;
        std::getline(file, version); // Getting the version line (e.g. 4.1 0 8)
        gmshFormat = std::stoul( version.substr( 0, version.find(" ")) ); // Retrieving the mesh format, keeping only the integer part
        std::getline(file, cmd); // $EndMeshFormat

        if (cmd.length() < 14 || cmd.substr(0, 14) != string("$EndMeshFormat")) // it should end with "$EndMeshFormat" or "$EndMeshFormat\r"
        {
            msg_error() << "No $EndMeshFormat flag found at the end of the file. Closing File";
            file.close();
            return false;
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
                    msg_error() << "End of file reached without finding the $Nodes section expected in MSH file format. Closing file.";
                    file.close();
                    return false;
                }
            }
        }
    }
    else if (cmd.length() >= 4 && cmd.substr(0, 4) == "$NOD")
    {
        // Legacy MSh format version 1 directly starts with the Nodes section
        // https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-1-_0028Legacy_0029
        // NB: corresponding line can be "$NOD", "$NOD\r"
        gmshFormat = 1;
    }
    else // If the first line is neither "$MeshFormat" or "$NOD", then the file is not in a registered MSH format
    {
        msg_error() << "File '" << d_filename << "' finally appears not to be a Gmsh file (first line doesn't match known formats).";
        file.close();
        return false;
    }

    std::istringstream nodeReader(cmd);
    string node;
    nodeReader >> node;
    // -- Reading file

    // By default for Gmsh file format, create subElements except if specified not to.
    if (!d_createSubelements.isSet())
        d_createSubelements.setValue(true);

    // TODO 2018-04-06: temporary change to unify loader API
    //fileRead = readGmsh(file, gmshFormat);
    (void)gmshFormat;
    file.close();
    helper::io::Mesh* _mesh = helper::io::Mesh::Create("gmsh", filename);

    copyMeshToData(*_mesh);
    delete _mesh;
    return true;
}


void MeshGmshLoader::doClearBuffers()
{
    /// Nothing to do if no output is added to the "filename" dataTrackerEngine.
}

void MeshGmshLoader::addInGroup(type::vector< sofa::core::loader::PrimitiveGroup>& group,int tag,int /*eid*/) {
    for (unsigned i=0;i<group.size();i++) {
        if (tag == group[i].p0) {
            group[i].nbp++;
            return;
        }
    }

    stringstream ss;
    const string s;
    ss << tag;

    group.push_back(sofa::core::loader::PrimitiveGroup(tag,1,s,s,-1));
}

void MeshGmshLoader::normalizeGroup(type::vector< sofa::core::loader::PrimitiveGroup>& group) {
    int start = 0;
    for (unsigned i=0;i<group.size();i++) {
        group[i].p0 = start;
        start += group[i].nbp;
    }
}

bool MeshGmshLoader::readGmsh(std::ifstream &file, const unsigned int gmshFormat)
{
    dmsg_info() << "Reading Gmsh file: " << gmshFormat;

    string cmd;

    unsigned int npoints = 0;
    unsigned int nelems = 0;

    // Accessors to complete the loader data
    auto my_positions = getWriteOnlyAccessor(d_positions);

    auto my_edges = getWriteOnlyAccessor(d_edges);
    auto my_triangles = getWriteOnlyAccessor(d_triangles);
    auto my_quads = getWriteOnlyAccessor(d_quads);
    auto my_tetrahedra = getWriteOnlyAccessor(d_tetrahedra);
    auto my_hexahedra = getWriteOnlyAccessor(d_hexahedra);

    auto my_highOrderEdgePositions = getWriteOnlyAccessor(d_highOrderEdgePositions);

    auto my_edgesGroups = getWriteOnlyAccessor(d_edgesGroups);
    auto my_trianglesGroups = getWriteOnlyAccessor(d_trianglesGroups);
    auto my_tetrahedraGroups = getWriteOnlyAccessor(d_tetrahedraGroups);
    auto my_hexahedraGroups = getWriteOnlyAccessor(d_hexahedraGroups);

    if (gmshFormat <= 2)
    {
        // --- Loading Vertices ---
        file >> npoints; //nb points

        std::vector<unsigned int> pmap; // map for reordering vertices possibly not well sorted
        for (unsigned int i = 0; i < npoints; ++i)
        {
            unsigned int index = i;
            double x, y, z;
            file >> index >> x >> y >> z;

            my_positions.push_back(Vec3(x, y, z));

            if (pmap.size() <= index)
                pmap.resize(index + 1);

            pmap[index] = i; // In case of hole or switch
        }

        file >> cmd;
        if (cmd.length() < 7 || cmd.substr(0, 7) != "$ENDNOD") // can be "$ENDNOD" or "$ENDNOD\r"
        {
            if (cmd.length() < 9 || cmd.substr(0, 9) != "$EndNodes") // can be "$EndNodes" or "$EndNodes\r"
            {
                msg_error() << "'$ENDNOD' or '$EndNodes' expected, found '" << cmd << "'";
                file.close();
                return false;
            }
        }


        // --- Loading Elements ---
        file >> cmd;
        if (cmd.length() < 4 || cmd.substr(0, 4) != "$ELM") // can be "$ELM" or "$ELM\r"
        {
            if (cmd.length() < 9 || cmd.substr(0, 9) != "$Elements") // can be "$ELM" or "$ELM\r"
            {
                msg_error() << "'$ELM' or '$Elements' expected, found '" << cmd << "'";
                file.close();
                return false;
            }
        }

        file >> nelems; //Loading number of Element

        for (unsigned int i = 0; i < nelems; ++i) // for each elem
        {
            int index, etype, rphys, relem, nnodes, ntags, tag = 0; // TODO: i don't know if tag must be set to 0, but if it's not, the application assert / crash on Windows (uninitialized value)

            if (gmshFormat == 1)
            {
                // version 1.0 format is
                // elm-number elm-type reg-phys reg-elem number-of-nodes <node-number-list ...>
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
                    msg_warning() << "Elements of type 1, 2, 3, 4, 5, or 6 expected. Element of type " << etype << " found.";
                    nnodes = 0;
                }
            }


            //store real index of node and not line index
            type::vector <unsigned int> nodes;
            nodes.resize(nnodes);
            const unsigned int edgesInQuadraticTriangle[3][2] = { {0,1}, {1,2}, {2,0} };
            const unsigned int edgesInQuadraticTetrahedron[6][2] = { {0,1}, {1,2}, {0,2},{0,3},{2,3},{1,3} };
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
                addInGroup(my_edgesGroups.wref(), tag, my_edges.size());
                addEdge(my_edges.wref(), Edge(nodes[0], nodes[1]));
                break;
            case 2: // Triangle
                addInGroup(my_trianglesGroups.wref(), tag, my_triangles.size());
                addTriangle(my_triangles.wref(), Triangle(nodes[0], nodes[1], nodes[2]));
                break;
            case 3: // Quad
                addQuad(my_quads.wref(), Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
                break;
            case 4: // Tetra
                addInGroup(my_tetrahedraGroups.wref(), tag, my_tetrahedra.size());
                addTetrahedron(my_tetrahedra.wref(), Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                break;
            case 5: // Hexa
                addInGroup(my_hexahedraGroups.wref(), tag, my_hexahedra.size());
                addHexahedron(my_hexahedra.wref(), Hexahedron(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]));
                break;
            case 8: // quadratic edge
                addInGroup(my_edgesGroups.wref(), tag, my_edges.size());
                addEdge(my_edges.wref(), Edge(nodes[0], nodes[1]));
                {
                    HighOrderEdgePosition hoep;
                    hoep[0] = nodes[2];
                    hoep[1] = my_edges.size() - 1;
                    hoep[2] = 1;
                    hoep[3] = 1;
                    my_highOrderEdgePositions.push_back(hoep);
                }
                break;
            case 9: // quadratic triangle
                addInGroup(my_trianglesGroups.wref(), tag, my_triangles.size());
                addTriangle(my_triangles.wref(), Triangle(nodes[0], nodes[1], nodes[2]));
                {
                    HighOrderEdgePosition hoep;
                    for (j = 0; j < 3; ++j) {
                        size_t v0 = std::min(nodes[edgesInQuadraticTriangle[j][0]],
                            nodes[edgesInQuadraticTriangle[j][1]]);
                        size_t v1 = std::max(nodes[edgesInQuadraticTriangle[j][0]],
                            nodes[edgesInQuadraticTriangle[j][1]]);
                        Edge e(v0, v1);
                        if (edgeSet.find(e) == edgeSet.end()) {
                            edgeSet.insert(e);
                            addEdge(my_edges.wref(), v0, v1);
                            hoep[0] = nodes[j + 3];
                            hoep[1] = my_edges.size() - 1;
                            hoep[2] = 1;
                            hoep[3] = 1;
                            my_highOrderEdgePositions.push_back(hoep);
                        }
                    }
                }
                break;
            case 11: // quadratic tetrahedron
                addInGroup(my_tetrahedraGroups.wref(), tag, my_tetrahedra.size());
                addTetrahedron(my_tetrahedra.wref(), Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                {
                    HighOrderEdgePosition hoep;
                    for (j = 0; j < 6; ++j) {
                        size_t v0 = std::min(nodes[edgesInQuadraticTetrahedron[j][0]],
                            nodes[edgesInQuadraticTetrahedron[j][1]]);
                        size_t v1 = std::max(nodes[edgesInQuadraticTetrahedron[j][0]],
                            nodes[edgesInQuadraticTetrahedron[j][1]]);
                        Edge e(v0, v1);
                        if (edgeSet.find(e) == edgeSet.end()) {
                            edgeSet.insert(e);
                            addEdge(my_edges.wref(), v0, v1);
                            hoep[0] = nodes[j + 4];
                            hoep[1] = my_edges.size() - 1;
                            hoep[2] = 1;
                            hoep[3] = 1;
                            my_highOrderEdgePositions.push_back(hoep);
                        }
                    }
                }
                break;
            default:
                //if the type is not handled, skip rest of the line
                string tmp;
                std::getline(file, tmp);
            }
        }
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
                my_positions.push_back(Vec3(x, y, z));
            }
        }

        std::getline(file, cmd);
        if (cmd != "$EndNodes")
        {
            msg_error("MeshGmshLoader") << "'$EndNodes' expected, found '" << cmd << "'";
            return false;
        }

        // --- Parsing the $Elements section --- //

        std::getline(file, cmd);
        if (cmd != "$Elements")
        {
            msg_error("MeshGmshLoader") << "'$Elements' expected, found '" << cmd << "'";
            return false;
        }

        std::getline(file, cmd); // Getting first line of $Elements
        std::istringstream elementsHeader(cmd);
        unsigned int nbElements, minElementTag, maxElementTag;
        elementsHeader >> nbEntityBlocks >> nbElements >> minElementTag >> maxElementTag;

        // Common information to add second order triangles (elementType = 9) and tetrahedra (elementType = 11)
        const unsigned int edgesInQuadraticTriangle[3][2] = { {0,1}, {1,2}, {2,0} };
        const unsigned int edgesInQuadraticTetrahedron[6][2] = { {0,1}, {1,2}, {0,2},{0,3},{2,3},{1,3} };
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
                msg_error("MeshGmshLoader") << "Elements of type 1, 2, 3, 4, 5, 8, 9, 11 or 15 expected. Element of type " << elementType << " found.";
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
                    nodes[i] = nodeId - 1; //To account for the fact that node indices in the MSH file format start with 1 instead of 0
                }

                switch (elementType)
                {
                case 1: // Line
                    addInGroup(my_edgesGroups.wref(), elementTag, my_edges.size());
                    addEdge(my_edges.wref(), Edge(nodes[0], nodes[1]));
                    break;
                case 2: // Triangle
                    addInGroup(my_trianglesGroups.wref(), elementTag, my_triangles.size());
                    addTriangle(my_triangles.wref(), Triangle(nodes[0], nodes[1], nodes[2]));
                    break;
                case 3: // Quadrangle
                    addQuad(my_quads.wref(), Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
                    break;
                case 4: // Tetrahedron
                    addInGroup(my_tetrahedraGroups.wref(), elementTag, my_tetrahedra.size());
                    addTetrahedron(my_tetrahedra.wref(), Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                    break;
                case 5: // Hexahedron
                    addInGroup(my_hexahedraGroups.wref(), elementTag, my_hexahedra.size());
                    addHexahedron(my_hexahedra.wref(), Hexahedron(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]));
                    break;
                case 8: // Second order line
                    addInGroup(my_edgesGroups.wref(), elementTag, my_edges.size());
                    addEdge(my_edges.wref(), Edge(nodes[0], nodes[1]));
                    {
                        HighOrderEdgePosition hoep;
                        hoep[0] = nodes[2];
                        hoep[1] = my_edges.size() - 1;
                        hoep[2] = 1;
                        hoep[3] = 1;
                        my_highOrderEdgePositions.push_back(hoep);
                    }
                    break;
                case 9: // Second order triangle
                    addInGroup(my_trianglesGroups.wref(), elementTag, my_triangles.size());
                    addTriangle(my_triangles.wref(), Triangle(nodes[0], nodes[1], nodes[2]));
                    {
                        HighOrderEdgePosition hoep;
                        for (size_t j = 0; j < 3; ++j)
                        {
                            size_t v0 = std::min(nodes[edgesInQuadraticTriangle[j][0]],
                                nodes[edgesInQuadraticTriangle[j][1]]);
                            size_t v1 = std::max(nodes[edgesInQuadraticTriangle[j][0]],
                                nodes[edgesInQuadraticTriangle[j][1]]);
                            Edge e(v0, v1);
                            if (edgeSet.find(e) == edgeSet.end())
                            {
                                edgeSet.insert(e);
                                addEdge(my_edges.wref(), v0, v1);
                                hoep[0] = nodes[j + 3];
                                hoep[1] = my_edges.size() - 1;
                                hoep[2] = 1;
                                hoep[3] = 1;
                                my_highOrderEdgePositions.push_back(hoep);
                            }
                        }
                    }
                    break;
                case 11: // Second order tetrahedron
                    addInGroup(my_tetrahedraGroups.wref(), elementTag, my_tetrahedra.size());
                    addTetrahedron(my_tetrahedra.wref(), Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3]));
                    {
                        HighOrderEdgePosition hoep;
                        for (size_t j = 0; j < 6; ++j)
                        {
                            size_t v0 = std::min(nodes[edgesInQuadraticTetrahedron[j][0]],
                                nodes[edgesInQuadraticTetrahedron[j][1]]);
                            size_t v1 = std::max(nodes[edgesInQuadraticTetrahedron[j][0]],
                                nodes[edgesInQuadraticTetrahedron[j][1]]);
                            Edge e(v0, v1);
                            if (edgeSet.find(e) == edgeSet.end())
                            {
                                edgeSet.insert(e);
                                addEdge(my_edges.wref(), v0, v1);
                                hoep[0] = nodes[j + 4];
                                hoep[1] = my_edges.size() - 1;
                                hoep[2] = 1;
                                hoep[3] = 1;
                                my_highOrderEdgePositions.push_back(hoep);
                            }
                        }
                    }
                    break;
                    // default: if the type is not handled, nothing to be done
                }
            } // end of loop over the elements in one entity block
        } //end of loop over the entity blocks
    }

    normalizeGroup(my_edgesGroups.wref());
    normalizeGroup(my_trianglesGroups.wref());
    normalizeGroup(my_tetrahedraGroups.wref());
    normalizeGroup(my_hexahedraGroups.wref());

    file >> cmd;
    if (cmd != "$ENDELM" && cmd != "$EndElements")
    {
        msg_error("MeshGmshLoader") << "'$ENDELM' or '$EndElements' expected, found '" << cmd << "'";
        file.close();
        return false;
    }

    file.close();
    return true;
}


} //namespace sofa::component::io::mesh
