/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <SofaBaseTopology/GridTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(GridTopology)

int GridTopologyClass = core::RegisterObject("Base class fo a regular grid in 3D")
        .addAlias("Grid")
        .add< GridTopology >()
        ;


GridTopology::GridUpdate::GridUpdate(GridTopology *t):
    topology(t)
{
    addInput(&t->d_n);
    addOutput(&t->seqEdges);
    addOutput(&t->seqQuads);
    addOutput(&t->seqHexahedra);
    setDirtyValue();
}

void GridTopology::GridUpdate::update()
{
    updateEdges();
    updateQuads();
    updateHexas();
}

void GridTopology::GridUpdate::updateEdges()
{
    SeqEdges& edges = *topology->seqEdges.beginEdit();
    const Vec3i& n = topology->d_n.getValue();
    edges.clear();
    edges.reserve( (n[0]-1)*n[1]*n[2] +
            n[0]*(n[1]-1)*n[2] +
            n[0]*n[1]*(n[2]-1) );
    // lines along X
    for (int z=0; z<n[2]; z++)
        for (int y=0; y<n[1]; y++)
            for (int x=0; x<n[0]-1; x++)
                edges.push_back(Edge(topology->point(x,y,z),topology->point(x+1,y,z)));
    // lines along Y
    for (int z=0; z<n[2]; z++)
        for (int y=0; y<n[1]-1; y++)
            for (int x=0; x<n[0]; x++)
                edges.push_back(Edge(topology->point(x,y,z),topology->point(x,y+1,z)));
    // lines along Z
    for (int z=0; z<n[2]-1; z++)
        for (int y=0; y<n[1]; y++)
            for (int x=0; x<n[0]; x++)
                edges.push_back(Edge(topology->point(x,y,z),topology->point(x,y,z+1)));
    topology->seqEdges.endEdit();
}

void GridTopology::GridUpdate::updateQuads()
{
    SeqQuads& quads = *topology->seqQuads.beginEdit();
    const Vec3i& n = topology->d_n.getValue();
    quads.clear();
    quads.reserve((n[0]-1)*(n[1]-1)*n[2]+(n[0]-1)*n[1]*(n[2]-1)+n[0]*(n[1]-1)*(n[2]-1));
    // quads along XY plane
    for (int z=0; z<n[2]; z++)
        for (int y=0; y<n[1]-1; y++)
            for (int x=0; x<n[0]-1; x++)
                quads.push_back(Quad(topology->point(x,y,z),
                        topology->point(x+1,y,z),
                        topology->point(x+1,y+1,z),
                        topology->point(x,y+1,z)));
    // quads along XZ plane
    for (int z=0; z<n[2]-1; z++)
        for (int y=0; y<n[1]; y++)
            for (int x=0; x<n[0]-1; x++)
                quads.push_back(Quad(topology->point(x,y,z),
                        topology->point(x+1,y,z),
                        topology->point(x+1,y,z+1),
                        topology->point(x,y,z+1)));
    // quads along YZ plane
    for (int z=0; z<n[2]-1; z++)
        for (int y=0; y<n[1]-1; y++)
            for (int x=0; x<n[0]; x++)
                quads.push_back(Quad(topology->point(x,y,z),
                        topology->point(x,y+1,z),
                        topology->point(x,y+1,z+1),
                        topology->point(x,y,z+1)));

    topology->seqQuads.endEdit();
}

void GridTopology::GridUpdate::updateHexas()
{
    SeqHexahedra& hexahedra = *topology->seqHexahedra.beginEdit();
    const Vec3i& n = topology->d_n.getValue();
    hexahedra.clear();
    hexahedra.reserve((n[0]-1)*(n[1]-1)*(n[2]-1));
    for (int z=0; z<n[2]-1; z++)
        for (int y=0; y<n[1]-1; y++)
            for (int x=0; x<n[0]-1; x++)
#ifdef SOFA_NEW_HEXA
                hexahedra.push_back(Hexa(topology->point(x  ,y  ,z  ),topology->point(x+1,y  ,z  ),
                        topology->point(x+1,y+1,z  ),topology->point(x  ,y+1,z  ),
                        topology->point(x  ,y  ,z+1),topology->point(x+1,y  ,z+1),
                        topology->point(x+1,y+1,z+1),topology->point(x  ,y+1,z+1)));
#else
                hexahedra.push_back(Hexa(topology->point(x  ,y  ,z  ),topology->point(x+1,y  ,z  ),
                        topology->point(x  ,y+1,z  ),topology->point(x+1,y+1,z  ),
                        topology->point(x  ,y  ,z+1),topology->point(x+1,y  ,z+1),
                        topology->point(x  ,y+1,z+1),topology->point(x+1,y+1,z+1)));
#endif
    topology->seqHexahedra.endEdit();
}

GridTopology::GridTopology()
    : d_n(initData(&d_n,Vec3i(2,2,2),"n","grid resolution"))
    , d_createTexCoords(initData(&d_createTexCoords, (bool)false, "createTexCoords", "If set to true, virtual texture coordinates will be generated using 3D interpolation."))
{
    setSize();
    GridUpdate::SPtr gridUpdate = sofa::core::objectmodel::New<GridUpdate>(this);
    this->addSlave(gridUpdate);
}

GridTopology::GridTopology(int _nx, int _ny, int _nz)
    : d_n(initData(&d_n,Vec3i(_nx,_ny,_nz),"n","grid resolution"))
    , d_createTexCoords(initData(&d_createTexCoords, (bool)false, "createTexCoords", "If set to true, virtual texture coordinates will be generated using 3D interpolation."))
{
    nbPoints = _nx*_ny*_nz;
    this->d_n.setValue(Vec3i(_nx,_ny,_nz));
}

GridTopology::GridTopology( Vec3i np )
    : d_n(initData(&d_n,np,"n","grid resolution"))
    , d_createTexCoords(initData(&d_createTexCoords, (bool)false, "createTexCoords", "If set to true, virtual texture coordinates will be generated using 3D interpolation."))
{
    nbPoints = np[0]*np[1]*np[2];
    this->d_n.setValue(np);
}

void GridTopology::init()
{
    if (d_createTexCoords.getValue())
        this->createTexCoords();

    this->reinit();
}

void GridTopology::setSize(int nx, int ny, int nz)
{
//    std::cerr<<"GridTopology::setSize(int nx, int ny, int nz), n = "<< n.getValue() << std::endl;
    if (nx == this->d_n.getValue()[0] && ny == this->d_n.getValue()[1] && nz == this->d_n.getValue()[2])
        return;
    this->d_n.setValue(Vec3i(nx,ny,nz));
    setSize();
}

void GridTopology::setNumVertices(int nx, int ny, int nz)
{
    setSize(nx,ny,nz);
}

void GridTopology::setNumVertices(Vec3i n)
{
    setSize(n[0],n[1],n[2]);
}

void GridTopology::setSize()
{
    this->nbPoints = d_n.getValue()[0]*d_n.getValue()[1]*d_n.getValue()[2];
}


GridTopology::Hexa GridTopology::getHexaCopy(int i)
{
    int x = i%(d_n.getValue()[0]-1); i/=(d_n.getValue()[0]-1);
    int y = i%(d_n.getValue()[1]-1); i/=(d_n.getValue()[1]-1);
    int z = i;
    return getHexahedron(x,y,z);
}

GridTopology::Hexa GridTopology::getHexahedron(int x, int y, int z)
{
#ifdef SOFA_NEW_HEXA
    return Hexa(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
            point(x+1,y+1,z  ),point(x  ,y+1,z  ),
            point(x  ,y  ,z+1),point(x+1,y  ,z+1),
            point(x+1,y+1,z+1),point(x  ,y+1,z+1));
#else
    return Hexa(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
            point(x  ,y+1,z  ),point(x+1,y+1,z  ),
            point(x  ,y  ,z+1),point(x+1,y  ,z+1),
            point(x  ,y+1,z+1),point(x+1,y+1,z+1));
#endif
}

GridTopology::Quad GridTopology::getQuadCopy(int i)
{
    if (d_n.getValue()[0] == 1)
    {
        int y = i%(d_n.getValue()[1]-1);
        i/=(d_n.getValue()[1]-1);
        int z = i%(d_n.getValue()[2]-1);

        return getQuad(1,y,z);
    }
    else if (d_n.getValue()[1] == 1)
    {
        int x = i%(d_n.getValue()[0]-1);
        i/=(d_n.getValue()[0]-1);
        int z = i%(d_n.getValue()[2]-1);

        return getQuad(x,1,z);
    }
    else
    {
        int x = i%(d_n.getValue()[0]-1);
        i/=(d_n.getValue()[0]-1);
        int y = i%(d_n.getValue()[1]-1);

        return getQuad(x,y,1);
    }
}

GridTopology::Quad GridTopology::getQuad(int x, int y, int z)
{
    if (d_n.getValue()[2] == 1)
        return Quad(point(x, y, 1), point(x+1, y, 1),
                point(x+1, y+1, 1), point(x, y+1, 1));
    else if (d_n.getValue()[1] == 1)
        return Quad(point(x, 1, z), point(x+1, 1, z),
                point(x+1, 1, z+1), point(x, 1, z+1));
    else
        return Quad(point(1, y, z),point(1, y+1, z),
                point(1, y+1, z+1),point(1, y, z+1));
}

} // namespace topology

} // namespace component

} // namespace sofa

