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
    MeshTopologyLoader():m_mesh(NULL) {}
    virtual ~MeshTopologyLoader() {}
    bool load(const char *filename);
    virtual void setNbPoints(int /*n*/) {}
    virtual void setNbLines(int /*n*/) {}
    virtual void setNbEdges(int /*n*/) {}
    virtual void setNbTriangles(int /*n*/) {}
    virtual void setNbQuads(int /*n*/) {}
    virtual void setNbTetrahedra(int /*n*/) {}
    virtual void setNbCubes(int /*n*/) {}
    virtual void addPoint(SReal /*px*/, SReal /*py*/, SReal /*pz*/) {}
    virtual void addLine(int /*p1*/, int /*p2*/) {}
    virtual void addTriangle(int /*p1*/, int /*p2*/, int /*p3*/) {}
    virtual void addQuad(int /*p1*/, int /*p2*/, int /*p3*/, int /*p4*/) {}
    virtual void addTetra(int /*p1*/, int /*p2*/, int /*p3*/, int /*p4*/) {}
    virtual void addCube(int /*p1*/, int /*p2*/, int /*p3*/, int /*p4*/, int /*p5*/, int /*p6*/, int /*p7*/, int /*p8*/) {}
private:
    /// method will create a MeshObj which will parse the file. Then data are loaded into the current topology
    bool loadObj(const char *filename);

    /// method will create a MeshGmsh which will parse the file. Then will call @see addMeshtoTopology() to add mesh data into topology
    bool loadGmsh(const char *filename);

    /// method will create a MeshSTL which will parse the file. Then will call @see addMeshtoTopology() to add mesh data into topology
    bool loadStl(const char *filename);

    /// method will create a MeshXsp which will parse the file. Then will call @see addMeshtoTopology() to add mesh data into topology
    bool loadXsp(const char *filename);

    /// method called when format is not found. Will parse header and dispatch into @see loadGmsh or @see loadXsp or @see loadMesh
    bool loadMeshFile(const char *filename);
    

    bool loadVtk(const char *filename);
        
    /// method to load unknown format. TODO remove this method and deprecated attached format.
    bool loadMesh(std::ifstream &file);

    // will take all data from loaded into @see m_mesh and add it to the current topology using methods api.
    bool addMeshtoTopology();

private:
    helper::io::Mesh* m_mesh;
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
