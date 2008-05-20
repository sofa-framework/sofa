/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/topology/GridTopology.h>
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

GridTopology::GridTopology()
    : n(initData(&n,Vec<3, int>(0,0,0),"n","grid resolution"))
{
}

GridTopology::GridTopology(int _nx, int _ny, int _nz)
    : n(initData(&n,Vec<3, int>(_nx,_ny,_nz),"n","grid resolution"))
{
    nbPoints = _nx*_ny*_nz;
}

void GridTopology::setSize(int nx, int ny, int nz)
{
    if (nx == this->n.getValue()[0] && ny == this->n.getValue()[1] && nz == this->n.getValue()[2])
        return;
    this->n.setValue(Vec<3, int>(nx,ny,nz));
    setSize();
}

void GridTopology::setSize()
{
    this->nbPoints = n.getValue()[0]*n.getValue()[1]*n.getValue()[2];
    invalidate();
}

void GridTopology::updateEdges()
{
    SeqEdges& edges = *seqEdges.beginEdit();
    edges.clear();
    edges.reserve((n.getValue()[0]-1)*n.getValue()[1]*n.getValue()[2]+n.getValue()[0]*(n.getValue()[1]-1)*n.getValue()[2]+n.getValue()[0]*n.getValue()[1]*(n.getValue()[2]-1));
    // lines along X
    for (int z=0; z<n.getValue()[2]; z++)
        for (int y=0; y<n.getValue()[1]; y++)
            for (int x=0; x<n.getValue()[0]-1; x++)
                edges.push_back(Edge(point(x,y,z),point(x+1,y,z)));
    // lines along Y
    for (int z=0; z<n.getValue()[2]; z++)
        for (int y=0; y<n.getValue()[1]-1; y++)
            for (int x=0; x<n.getValue()[0]; x++)
                edges.push_back(Edge(point(x,y,z),point(x,y+1,z)));
    // lines along Z
    for (int z=0; z<n.getValue()[2]-1; z++)
        for (int y=0; y<n.getValue()[1]; y++)
            for (int x=0; x<n.getValue()[0]; x++)
                edges.push_back(Edge(point(x,y,z),point(x,y,z+1)));
    seqEdges.endEdit();
}

void GridTopology::updateQuads()
{
    seqQuads.clear();
    seqQuads.reserve((n.getValue()[0]-1)*(n.getValue()[1]-1)*n.getValue()[2]+(n.getValue()[0]-1)*n.getValue()[1]*(n.getValue()[2]-1)+n.getValue()[0]*(n.getValue()[1]-1)*(n.getValue()[2]-1));
    // quads along XY plane
    for (int z=0; z<n.getValue()[2]; z++)
        for (int y=0; y<n.getValue()[1]-1; y++)
            for (int x=0; x<n.getValue()[0]-1; x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x+1,y,z),point(x+1,y+1,z),point(x,y+1,z)));
    // quads along XZ plane
    for (int z=0; z<n.getValue()[2]-1; z++)
        for (int y=0; y<n.getValue()[1]; y++)
            for (int x=0; x<n.getValue()[0]-1; x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x+1,y,z),point(x+1,y,z+1),point(x,y,z+1)));
    // quads along YZ plane
    for (int z=0; z<n.getValue()[2]-1; z++)
        for (int y=0; y<n.getValue()[1]-1; y++)
            for (int x=0; x<n.getValue()[0]; x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x,y+1,z),point(x,y+1,z+1),point(x,y,z+1)));
}

void GridTopology::updateHexas()
{
    seqHexas.clear();
    seqHexas.reserve((n.getValue()[0]-1)*(n.getValue()[1]-1)*(n.getValue()[2]-1));
    for (int z=0; z<n.getValue()[2]-1; z++)
        for (int y=0; y<n.getValue()[1]-1; y++)
            for (int x=0; x<n.getValue()[0]-1; x++)
#ifdef SOFA_NEW_HEXA
                seqHexas.push_back(Hexa(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
                        point(x+1,y+1,z  ),point(x  ,y+1,z  ),
                        point(x  ,y  ,z+1),point(x+1,y  ,z+1),
                        point(x+1,y+1,z+1),point(x  ,y+1,z+1)));
#else
                seqHexas.push_back(Hexa(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
                        point(x  ,y+1,z  ),point(x+1,y+1,z  ),
                        point(x  ,y  ,z+1),point(x+1,y  ,z+1),
                        point(x  ,y+1,z+1),point(x+1,y+1,z+1)));
#endif
}

GridTopology::Hexa GridTopology::getHexaCopy(int i)
{
    int x = i%(n.getValue()[0]-1); i/=(n.getValue()[0]-1);
    int y = i%(n.getValue()[1]-1); i/=(n.getValue()[1]-1);
    int z = i;
    return getHexa(x,y,z);
}

GridTopology::Hexa GridTopology::getHexa(int x, int y, int z)
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
    if (n.getValue()[0] == 1)
    {
        int y = i%(n.getValue()[1]-1);
        i/=(n.getValue()[1]-1);
        int z = i%(n.getValue()[2]-1);

        return getQuad(1,y,z);
    }
    else if (n.getValue()[1] == 1)
    {
        int x = i%(n.getValue()[0]-1);
        i/=(n.getValue()[0]-1);
        int z = i%(n.getValue()[2]-1);

        return getQuad(x,1,z);
    }
    else
    {
        int x = i%(n.getValue()[0]-1);
        i/=(n.getValue()[0]-1);
        int y = i%(n.getValue()[1]-1);

        return getQuad(x,y,1);
    }
}

GridTopology::Quad GridTopology::getQuad(int x, int y, int z)
{
    if (n.getValue()[2] == 1)
        return Quad(point(x, y, 1), point(x+1, y, 1),
                point(x+1, y+1, 1), point(x, y+1, 1));
    else if (n.getValue()[1] == 1)
        return Quad(point(x, 1, z), point(x+1, 1, z),
                point(x+1, 1, z+1), point(x, 1, z+1));
    else
        return Quad(point(1, y, z),point(1, y+1, z),
                point(1, y+1, z+1),point(1, y, z+1));
}

} // namespace topology

} // namespace component

} // namespace sofa

