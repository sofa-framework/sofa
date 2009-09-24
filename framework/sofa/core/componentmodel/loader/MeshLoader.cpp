/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/componentmodel/loader/MeshLoader.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace loader
{

using namespace sofa::defaulttype;

MeshLoader::MeshLoader() : BaseLoader()
    , positions(initData(&positions,"position","Vertices of the mesh loaded"))
    , edges(initData(&edges,"edges","Edges of the mesh loaded"))
    , triangles(initData(&triangles,"triangles","Triangles of the mesh loaded"))
    , quads(initData(&quads,"quads","Quads of the mesh loaded"))
    , polygons(initData(&polygons,"polygons","Polygons of the mesh loaded"))
    , tetrahedra(initData(&tetrahedra,"tetrahedra","Tetrahedra of the mesh loaded"))
    , hexahedra(initData(&hexahedra,"hexahedra","Hexahedra of the mesh loaded"))
    , flipNormals(initData(&flipNormals, false,"flipNormals","Flip Normals"))
    //, triangulate(initData(&triangulate,false,"triangulate","Divide all polygons into triangles"))
    //, fillMState(initData(&fillMState,true,"fillMState","Must this mesh loader fill the mstate instead of manually or by using the topology"))
    //, facets(initData(&facets,"facets","Facets of the mesh loaded"))
{
    addAlias(&tetrahedra,"tetras");
    addAlias(&hexahedra,"hexas");
    //TODO: check if necessary!
    positions.setPersistent(false);
    edges.setPersistent(false);
    triangles.setPersistent(false);
    quads.setPersistent(false);
    polygons.setPersistent(false);
    tetrahedra.setPersistent(false);
    hexahedra.setPersistent(false);
}


void MeshLoader::init()
{
    std::cout << "MeshLoader::init()" << std::endl;

    if (canLoad())
        load(/*m_filename.getFullPath().c_str()*/);
    else
        std::cout << "Doing nothing" << std::endl;
}


bool MeshLoader::canLoad()
{
    FILE* file;
    char cmd[1024];

    // -- Check filename field:
    if(m_filename.getValue() == "")
    {
        std::cerr << "Error: MeshLoader: No file name given." << std::endl;
        return false;
    }

    // -- Check if file exist:
    const char* filename = m_filename.getFullPath().c_str();
    std::string sfilename (filename);

    if (!sofa::helper::system::DataRepository.findFile(sfilename))
    {
        std::cerr << "Error: MeshLoader: File '" << m_filename << "' not found. " << std::endl;
        return false;
    }

    // -- Check if file is readable:
    if ((file = fopen(filename, "r")) == NULL)
    {
        std::cerr << "Error: MeshLoader: Cannot read file '" << m_filename << "'." << std::endl;
        return false;
    }

    // -- Step 2.2: Check first line.
    if (!readLine(cmd, sizeof(cmd), file))
    {
        std::cerr << "Error: MeshLoader: Cannot read first line in file '" << m_filename << "'." << std::endl;
        fclose(file);
        return false;
    }

    fclose(file);
    return true;
}

void MeshLoader::addEdge(helper::vector<helper::fixed_array <unsigned int,2> >* pEdges, const helper::fixed_array <unsigned int,2> &p)
{
    pEdges->push_back(p);
}

void MeshLoader::addEdge(helper::vector<helper::fixed_array <unsigned int,2> >* pEdges, unsigned int p0, unsigned int p1)
{
    addEdge(pEdges, helper::fixed_array <unsigned int,2>(p0, p1));
}

void MeshLoader::addTriangle(helper::vector<helper::fixed_array <unsigned int,3> >* pTriangles, const helper::fixed_array <unsigned int,3> &p)
{
    if (flipNormals.getValue())
    {
        helper::fixed_array <unsigned int,3> revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pTriangles->push_back(revertP);
    }
    else
        pTriangles->push_back(p);
}

void MeshLoader::addTriangle(helper::vector<helper::fixed_array <unsigned int,3> >* pTriangles, unsigned int p0, unsigned int p1, unsigned int p2)
{
    addTriangle(pTriangles, helper::fixed_array <unsigned int,3>(p0, p1, p2));
}

void MeshLoader::addQuad(helper::vector<helper::fixed_array <unsigned int,4> >* pQuads, const helper::fixed_array <unsigned int,4> &p)
{
    if (flipNormals.getValue())
    {
        helper::fixed_array <unsigned int,4> revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pQuads->push_back(revertP);
    }
    else
        pQuads->push_back(p);
}

void MeshLoader::addQuad(helper::vector<helper::fixed_array <unsigned int,4> >* pQuads, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3)
{
    addQuad(pQuads, helper::fixed_array <unsigned int,4>(p0, p1, p2, p3));
}

void MeshLoader::addPolygon(helper::vector< helper::vector <unsigned int> >* pPolygons, const helper::vector<unsigned int> &p)
{
    if (flipNormals.getValue())
    {
        helper::vector<unsigned int> revertP(p.size());
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pPolygons->push_back(revertP);
    }
    else
        pPolygons->push_back(p);
}


void MeshLoader::addTetrahedron(helper::vector< helper::fixed_array<unsigned int,4> >* pTetrahedra, const helper::fixed_array<unsigned int,4> &p)
{
    pTetrahedra->push_back(p);
}

void MeshLoader::addTetrahedron(helper::vector< helper::fixed_array<unsigned int,4> >* pTetrahedra, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3)
{
    addTetrahedron(pTetrahedra, helper::fixed_array <unsigned int,4>(p0, p1, p2, p3));
}

void MeshLoader::addHexahedron(helper::vector< helper::fixed_array<unsigned int,8> >* pHexahedra,
        unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
        unsigned int p4, unsigned int p5, unsigned int p6, unsigned int p7)
{
    addHexahedron(pHexahedra, helper::fixed_array <unsigned int,8>(p0, p1, p2, p3, p4, p5, p6, p7));
}

void MeshLoader::addHexahedron(helper::vector< helper::fixed_array<unsigned int,8> >* pHexahedra, const helper::fixed_array<unsigned int,8> &p)
{
    pHexahedra->push_back(p);
}


} // namespace loader

} // namespace componentmodel

} // namespace core

} // namespace sofa

