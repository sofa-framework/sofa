/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
    MeshTopologyLoader():mesh(NULL) {}
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
    bool loadObj(const char *filename);
    bool loadMeshFile(const char *filename);
    bool loadVtk(const char *filename);
    bool loadStl(const char *filename);

    bool loadGmsh(std::ifstream &file, const int);
    bool loadXsp(std::ifstream &file, bool);
    bool loadMesh(std::ifstream &file);
    bool loadCGAL(const char *filename);

protected:
    helper::io::Mesh* mesh;
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
