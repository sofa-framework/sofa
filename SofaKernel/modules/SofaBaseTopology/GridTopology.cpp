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
#include <SofaBaseTopology/GridTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec.h>

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
    updateTriangles();
    updateHexas();
}

void GridTopology::GridUpdate::updateEdges()
{
    SeqEdges& edges = *topology->seqEdges.beginWriteOnly();
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

void GridTopology::GridUpdate::updateTriangles()
{
    // base on quads
    const SeqQuads& quads = topology->seqQuads.getValue();
    SeqTriangles& triangles = *topology->seqTriangles.beginWriteOnly();
    triangles.clear();
    triangles.reserve(quads.size()*2);

    for (unsigned int i=0; i<quads.size(); ++i)
    {
        triangles.push_back(Triangle(quads[i][0], quads[i][1], quads[i][2]));
        triangles.push_back(Triangle(quads[i][0], quads[i][2], quads[i][3]));
    }

    topology->seqTriangles.endEdit();
}

void GridTopology::GridUpdate::updateQuads()
{
    SeqQuads& quads = *topology->seqQuads.beginWriteOnly();
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
    SeqHexahedra& hexahedra = *topology->seqHexahedra.beginWriteOnly();
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

/// To avoid duplicating the code in the different variants of the constructor
/// this object is using the delegating constructor feature of c++ x11.
/// The following constructor is "chained" by the other constructors to
/// defined only one the member initialization.
GridTopology::GridTopology()
    : d_n(initData(&d_n,Vec3i(2,2,2),"n","grid resolution. (default = 2 2 2)"))
    , d_computeHexaList(initData(&d_computeHexaList, true, "computeHexaList", "put true if the list of Hexahedra is needed during init (default=true)"))
    , d_computeQuadList(initData(&d_computeQuadList, true, "computeQuadList", "put true if the list of Quad is needed during init (default=true)"))
    , d_computeEdgeList(initData(&d_computeEdgeList, true, "computeEdgeList", "put true if the list of Lines is needed during init (default=true)"))
    , d_computePointList(initData(&d_computePointList, true, "computePointList", "put true if the list of Points is needed during init (default=true)"))
    , d_createTexCoords(initData(&d_createTexCoords, (bool)false, "createTexCoords", "If set to true, virtual texture coordinates will be generated using 3D interpolation (default=false)."))
{
    setNbGridPoints();
    GridUpdate::SPtr gridUpdate = sofa::core::objectmodel::New<GridUpdate>(this);
    this->addSlave(gridUpdate);
}

/// This constructor is chained with the one without parameter
GridTopology::GridTopology(const Vec3i& dimXYZ ) :
    GridTopology()
{
    d_n.setValue(dimXYZ);
    checkGridResolution();
}

/// This constructor is chained with the one with a Vec3i parameter
GridTopology::GridTopology(int nx, int ny, int nz) :
    GridTopology(Vec3i(nx,ny,nz))
{
}

void GridTopology::init()
{
    // first check resolution
    checkGridResolution();

    if (d_computePointList.getValue())
        this->computePointList();

    if (d_createTexCoords.getValue())
        this->createTexCoords();

    if (d_computeHexaList.getValue())
        this->computeHexaList();

    if (d_computeQuadList.getValue())
        this->computeQuadList();

    if (d_computeEdgeList.getValue())
        this->computeEdgeList();

    Inherit1::init();
}

void GridTopology::reinit()
{
    checkGridResolution();
}

void GridTopology::setSize(int nx, int ny, int nz)
{
    if (nx == this->d_n.getValue()[0] && ny == this->d_n.getValue()[1] && nz == this->d_n.getValue()[2])
        return;
    this->d_n.setValue(Vec3i(nx,ny,nz));
    setNbGridPoints();

    checkGridResolution();
}

void GridTopology::checkGridResolution()
{
    const Vec3i& _n = d_n.getValue();

    if (_n[0] < 1 || _n[1] < 1 || _n[2] < 1)
    {
        msg_warning() << "The grid resolution: ["<< _n[0] << " ; " << _n[1] << " ; " << _n[2] <<
                         "] is outside the validity range. At least a resolution of 1 is needed in each 3D direction."
                         " Continuing with default value=[2; 2; 2]."
                         " Set a valid grid resolution to remove this warning message.";

        this->d_n.setValue(Vec3i(2,2,2));
        changeGridResolutionPostProcess();
    }

    setNbGridPoints();
}

Grid_dimension GridTopology::getDimensions() const
{
	const Vec3i& _n = d_n.getValue();
	int dim = 0;
	for (int i = 0; i<3; i++)
		if (_n[i] > 1) 
			dim++;

	return (Grid_dimension)dim;
}

void GridTopology::setSize(Vec3i n)
{
    setSize(n[0],n[1],n[2]);
}

void GridTopology::setNbGridPoints()
{
    this->setNbPoints(d_n.getValue()[0]*d_n.getValue()[1]*d_n.getValue()[2]);
}


void GridTopology::computeHexaList()
{
    updateHexahedra();
}

void GridTopology::computeQuadList()
{
//    updateQuads();
//    const SeqQuads seq_quads= this->getQuads();
//    sout<<"Init: Number of Quads ="<<seq_quads.size()<<sendl;
}

void GridTopology::computeEdgeList()
{
    //updateEdges();
//    const SeqLines seq_l=this->getLines();
//    sout<<"Init: Number of Lines ="<<seq_l.size()<<sendl;
}

void GridTopology::computePointList()
{
    int nbPoints= this->getNbPoints();
    // put the result in seqPoints
    SeqPoints& seq_P= *(seqPoints.beginWriteOnly());
    seq_P.resize(nbPoints);

    for (int i=0; i<nbPoints; i++)
    {
        seq_P[i] = this->getPoint(i);
    }

    seqPoints.endEdit();
}

unsigned GridTopology::getIndex( int i, int j, int k ) const
{
    return d_n.getValue()[0]* ( d_n.getValue()[1]*k + j ) + i;
}


sofa::defaulttype::Vector3 GridTopology::getPoint(int i) const
{
    int x = i%d_n.getValue()[0]; i/=d_n.getValue()[0];
    int y = i%d_n.getValue()[1]; i/=d_n.getValue()[1];
    int z = i;

    return getPointInGrid(x,y,z);
}

sofa::defaulttype::Vector3 GridTopology::getPointInGrid(int i, int j, int k) const
{
    unsigned int id = this->getIndex(i, j, k);
    if (id < seqPoints.getValue().size())
        return seqPoints.getValue()[id];
    else
        return sofa::defaulttype::Vector3();
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

