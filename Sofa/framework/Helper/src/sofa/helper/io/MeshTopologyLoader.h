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
#ifndef SOFA_HELPER_IO_MESHTOPOLOGYLOADER_H
#define SOFA_HELPER_IO_MESHTOPOLOGYLOADER_H

#include <sofa/helper/io/Mesh.h>
#include <fstream>

namespace sofa
{

namespace helper
{

namespace io
{

class SOFA_HELPER_API MeshTopologyLoader
{
public:
    using Index = sofa::Index;

    MeshTopologyLoader():m_mesh(nullptr) {}
    virtual ~MeshTopologyLoader() {}
    bool load(const char *filename);
    virtual void setNbPoints(Index /*n*/) {}
    virtual void setNbLines(Index /*n*/) {}
    virtual void setNbEdges(Index /*n*/) {}
    virtual void setNbTriangles(Index /*n*/) {}
    virtual void setNbQuads(Index /*n*/) {}
    virtual void setNbTetrahedra(Index /*n*/) {}
    virtual void setNbCubes(Index /*n*/) {}
    virtual void addPoint(SReal /*px*/, SReal /*py*/, SReal /*pz*/) {}
    virtual void addLine(Index /*p1*/, Index /*p2*/) {}
    virtual void addTriangle(Index /*p1*/, Index /*p2*/, Index /*p3*/) {}
    virtual void addQuad(Index /*p1*/, Index /*p2*/, Index /*p3*/, Index /*p4*/) {}
    virtual void addTetra(Index /*p1*/, Index /*p2*/, Index /*p3*/, Index /*p4*/) {}
    virtual void addCube(Index /*p1*/, Index /*p2*/, Index /*p3*/, Index /*p4*/, Index /*p5*/, Index /*p6*/, Index /*p7*/, Index /*p8*/) {}
private:
    /// method will create a MeshObj which will parse the file. Then data are loaded into the current topology
    bool loadObj(const char *filename);

    /// method will create a MeshGmsh which will parse the file. Then will call @see addMeshtoTopology() to add mesh data into topology
    bool loadGmsh(const char *filename);
    

    bool loadVtk(const char *filename);

    SOFA_MESHTOPOLOGYLOADER_LOADMESHFUNCTION_DEPRECATED()
    bool loadMesh(std::ifstream &file);

    // will take all data from loaded into @see m_mesh and add it to the current topology using methods api.
    bool addMeshtoTopology();

    helper::io::Mesh* m_mesh;
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
