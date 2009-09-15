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




} // namespace loader

} // namespace componentmodel

} // namespace core

} // namespace sofa

