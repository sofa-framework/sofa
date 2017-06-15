/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaLoader/MeshObjLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/io/File.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/Locale.h>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;
using namespace sofa::helper::io;
using namespace sofa::core::loader;
using std::istream;
using helper::vector;
using sofa::defaulttype::Vector3;
using helper::SVector;
using helper::WriteAccessor;
using std::string;

SOFA_DECL_CLASS(MeshObjLoader)

int MeshObjLoaderClass = core::RegisterObject("Specific mesh loader for Obj file format.")
        .add< MeshObjLoader >()
        ;



MeshObjLoader::MeshObjLoader()
    : MeshLoader()
    , d_storeGroups(initData(&d_storeGroups,false,"storeGroups", "should sub-groups be stored?"))
{
    d_positions.setPersistent(false);
    d_normals.setPersistent(false);
    d_edges.setPersistent(false);
    d_triangles.setPersistent(false);
    d_quads.setPersistent(false);
    d_edgesGroups.setPersistent(false);
    d_trianglesGroups.setPersistent(false);
    d_quadsGroups.setPersistent(false);
}


MeshObjLoader::~MeshObjLoader()
{
}


bool MeshObjLoader::load()
{
    sout << "Loading OBJ file: " << m_filename << sendl;

    bool fileRead = false;

    // -- Loading file
    if (!canLoad())
        return false;

    const char* filename = m_filename.getFullPath().c_str();
    File file;
    file.open(filename);
    istream stream(file.streambuf());

    // -- Reading file
    fileRead = this->readOBJ (stream, filename);
    file.close();

    return fileRead;
}


bool MeshObjLoader::readOBJ (istream &stream, const char* filename)
{
    SOFA_UNUSED(filename);

    dmsg_info() << " readOBJ" ;

    vector<Vector3>& my_positions = *(d_positions.beginWriteOnly());
    vector<int> nodes, nIndices;

    vector<Vector3> my_normals;
    SVector< SVector <int> > my_faceList;
    SVector< SVector <int> > my_normalsList;

    vector<Edge >& my_edges = *(d_edges.beginWriteOnly());
    vector<Triangle >& my_triangles = *(d_triangles.beginWriteOnly());
    vector<Quad >& my_quads = *(d_quads.beginWriteOnly());

    //BUGFIX: clear pre-existing data before loading the stream
    my_positions.clear();
    my_edges.clear();
    my_triangles.clear();
    my_quads.clear();
    d_edgesGroups.beginWriteOnly()->clear(); d_edgesGroups.endEdit();
    d_trianglesGroups.beginWriteOnly()->clear(); d_trianglesGroups.endEdit();
    d_quadsGroups.beginWriteOnly()->clear(); d_quadsGroups.endEdit();

    int vtn[3];
    Vector3 result;
    WriteAccessor<Data<vector< PrimitiveGroup> > > my_faceGroups[NBFACETYPE] =
    {
        d_edgesGroups,
        d_trianglesGroups,
        d_quadsGroups
    };
    string curGroupName = "Default_Group";
    string curMaterialName;
    int curMaterialId = -1;
    int nbFaces[NBFACETYPE] = {0}; // number of edges, triangles, quads
    int groupF0[NBFACETYPE] = {0}; // first primitives indices in current group for edges, triangles, quads
    string line;
    while( std::getline(stream,line) )
    {
        if (line.empty()) continue;
        std::istringstream values(line);
        string token;

        values >> token;
        if (token == "#")
        {
            /* comment */
        }
        else if (token == "v")
        {
            /* vertex */
            values >> result[0] >> result[1] >> result[2];
            my_positions.push_back(Vector3(result[0],result[1], result[2]));
        }
        else if (token == "vn")
        {
            /* normal */
            values >> result[0] >> result[1] >> result[2];
            my_normals.push_back(Vector3(result[0],result[1], result[2]));
        }
        else if ((token == "usemtl" || token == "g") && d_storeGroups.getValue() )
        {
            // end of current group
            //curGroup.nbp = nbf - curGroup.p0;
            for (int ft = 0; ft < NBFACETYPE; ++ft)
                if (nbFaces[ft] > groupF0[ft])
                {
                    my_faceGroups[ft].push_back(PrimitiveGroup(groupF0[ft], nbFaces[ft]-groupF0[ft], curMaterialName, curGroupName, curMaterialId));
                    groupF0[ft] = nbFaces[ft];
                }
            if (token == "g")
            {
                curGroupName.clear();
                while (!values.eof())
                {
                    string g;
                    values >> g;
                    if (!curGroupName.empty())
                        curGroupName += " ";
                    curGroupName += g;
                }
            }
        }
        else if (token == "l" || token == "f")
        {
            /* face */
            nodes.clear();
            nIndices.clear();

            while (!values.eof())
            {
                string face;
                values >> face;
                if (face.empty()) continue;
                for (int j = 0; j < 3; j++)
                {
                    vtn[j] = -1;
                    string::size_type pos = face.find('/');
                    string tmp = face.substr(0, pos);
                    if (pos == string::npos)
                        face = "";
                    else
                    {
                        face = face.substr(pos + 1);
                    }

                    if (!tmp.empty())
                    {
                        vtn[j] = atoi(tmp.c_str());
                        if (vtn[j] >= 1)
                            vtn[j] -=1; // -1 because the numerotation begins at 1 and a vector begins at 0
                        else if (vtn[j] < 0)
                            vtn[j] += (j==0) ? my_positions.size() : (j==2) ? my_normals.size() : 0/*my_texCoords.size()*/;
                        else
                        {
                            serr << "Invalid index " << tmp << sendl;
                            vtn[j] = -1;
                        }
                    }
                }

                nodes.push_back(vtn[0]);
                nIndices.push_back(vtn[2]);
            }

            my_faceList.push_back(nodes);
            my_normalsList.push_back(nIndices);

            if (nodes.size() == 2) // Edge
            {
                if (nodes[0]<nodes[1])
                    addEdge(&my_edges, Edge(nodes[0], nodes[1]));
                else
                    addEdge(&my_edges, Edge(nodes[1], nodes[0]));
                ++nbFaces[MeshObjLoader::EDGE];
            }
            else if (nodes.size()==4 && !this->d_triangulate.getValue()) // Quad
            {
                addQuad(&my_quads, Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
                ++nbFaces[MeshObjLoader::QUAD];
            }
            else // Triangulate
            {
                for (size_t j=2; j<nodes.size(); j++)
                    addTriangle(&my_triangles, Triangle(nodes[0], nodes[j-1], nodes[j]));
                ++nbFaces[MeshObjLoader::TRIANGLE];
            }

        }
    }

    // end of current group
    if(  d_storeGroups.getValue() )
    {
        for (size_t ft = 0; ft < NBFACETYPE; ++ft)
            if (nbFaces[ft] > groupF0[ft])
            {
                my_faceGroups[ft].push_back(PrimitiveGroup(groupF0[ft], nbFaces[ft]-groupF0[ft], curMaterialName, curGroupName, curMaterialId));
                groupF0[ft] = nbFaces[ft];
            }
    }

    vector<Vector3>& vNormals   = *d_normals.beginWriteOnly();
    size_t vertexCount = my_positions.size();

    if( my_normals.size() > 0 )
    {
        vNormals.resize(vertexCount);
    }
    else
    {
        vNormals.resize(0);
    }
    for (size_t fi=0; fi<my_faceList.size(); ++fi)
    {
        const SVector<int>& nodes = my_faceList[fi];
        const SVector<int>& nIndices = my_normalsList[fi];

        for (size_t i = 0; i < nodes.size(); ++i)
        {
            unsigned int pi = nodes[i];
            unsigned int ni = nIndices[i];
            if (i >= vertexCount) continue;
            if (ni < my_normals.size())
                vNormals[pi] += my_normals[ni];
        }
    }
    for (size_t i=0; i<vNormals.size(); ++i)
    {
        vNormals[i].normalize();
    }

    d_edgesGroups.endEdit();
    d_trianglesGroups.endEdit();
    d_quadsGroups.endEdit();
    d_positions.endEdit();
    d_edges.endEdit();
    d_triangles.endEdit();
    d_quads.endEdit();
    d_normals.endEdit();
    return true;
}

} // namespace loader

} // namespace component

} // namespace sofa

