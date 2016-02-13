/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseTopology/RegularGridTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;



void RegularGridTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    float scale=1.0f;
    if (arg->getAttribute("scale")!=NULL)
    {
        scale = (float)atof(arg->getAttribute("scale"));
    }

    this->GridTopology::parse(arg);
    if (arg->getAttribute("xmin") != NULL &&
        arg->getAttribute("ymin") != NULL &&
        arg->getAttribute("zmin") != NULL &&
        arg->getAttribute("xmax") != NULL &&
        arg->getAttribute("ymax") != NULL &&
        arg->getAttribute("zmax") != NULL )
    {
        const char* xmin = arg->getAttribute("xmin");
        const char* ymin = arg->getAttribute("ymin");
        const char* zmin = arg->getAttribute("zmin");
        const char* xmax = arg->getAttribute("xmax");
        const char* ymax = arg->getAttribute("ymax");
        const char* zmax = arg->getAttribute("zmax");
        min.setValue(Vector3((SReal)atof(xmin)*scale, (SReal)atof(ymin)*scale, (SReal)atof(zmin)*scale));
        max.setValue(Vector3((SReal)atof(xmax)*scale, (SReal)atof(ymax)*scale, (SReal)atof(zmax)*scale));
    }
    this->setPos(min.getValue()[0],max.getValue()[0],min.getValue()[1],max.getValue()[1],min.getValue()[2],max.getValue()[2]);

}

SOFA_DECL_CLASS(RegularGridTopology)

int RegularGridTopologyClass = core::RegisterObject("Regular grid in 3D")
        .addAlias("RegularGrid")
        .add< RegularGridTopology >()
        ;

RegularGridTopology::RegularGridTopology()
    :
    computeHexaList(initData(&computeHexaList, true, "computeHexaList", "put true if the list of Hexahedra is needed during init")),
    computeQuadList(initData(&computeQuadList, true, "computeQuadList", "put true if the list of Quad is needed during init")),
    computeEdgeList(initData(&computeEdgeList, true, "computeEdgeList", "put true if the list of Lines is needed during init")),
    computePointList(initData(&computePointList, true, "computePointList", "put true if the list of Points is needed during init")),
    min(initData(&min,Vector3(0.0f,0.0f,0.0f),"min", "Min end of the diagonal")),
    max(initData(&max,Vector3(1.0f,1.0f,1.0f),"max", "Max end of the diagonal")),
    p0(initData(&p0,Vector3(0.0f,0.0f,0.0f),"p0", "Offset all the grid points")),
    _cellWidth(initData(&_cellWidth, (SReal)0.0, "cellWidth","if > 0 : dimension of each cell in the created grid. Otherwise, the cell size is computed based on min, max, and resolution n."))
{
}

RegularGridTopology::RegularGridTopology(Vec3i n, BoundingBox b)
    : GridTopology(n),
      computeHexaList(initData(&computeHexaList, true, "computeHexaList", "put true if the list of Hexahedra is needed during init")),
      //  computeTetraList(initData(&computeTetraList, false, "computeTetraList", "put true if the list of Tetrahedra is needed during init")),
      computeQuadList(initData(&computeQuadList, true, "computeQuadList", "put true if the list of Quad is needed during init")),
      //   computeTriList(initData(&computeTriList, false, "computeTriList", "put true if the list of Triangle is needed during init")),
      computeEdgeList(initData(&computeEdgeList, true, "computeEdgeList", "put true if the list of Lines is needed during init")),
      computePointList(initData(&computePointList, true, "computePointList", "put true if the list of Points is needed during init")),
      min(initData(&min,Vector3(0.0f,0.0f,0.0f),"min", "Min")),
      max(initData(&max,Vector3(1.0f,1.0f,1.0f),"max", "Max")),
      p0(initData(&p0,Vector3(0.0f,0.0f,0.0f),"p0", "p0")),
      _cellWidth(initData(&_cellWidth, (SReal)0.0, "cellWidth","if > 0 : dimension of each cell in the created grid"))

{
    setPos(b);
}

RegularGridTopology::RegularGridTopology(int nx, int ny, int nz)
    : GridTopology(nx, ny, nz),
      computeHexaList(initData(&computeHexaList, true, "computeHexaList", "put true if the list of Hexahedra is needed during init")),
      //  computeTetraList(initData(&computeTetraList, false, "computeTetraList", "put true if the list of Tetrahedra is needed during init")),
      computeQuadList(initData(&computeQuadList, true, "computeQuadList", "put true if the list of Quad is needed during init")),
      //   computeTriList(initData(&computeTriList, false, "computeTriList", "put true if the list of Triangle is needed during init")),
      computeEdgeList(initData(&computeEdgeList, true, "computeEdgeList", "put true if the list of Lines is needed during init")),
      computePointList(initData(&computePointList, true, "computePointList", "put true if the list of Points is needed during init")),
      min(initData(&min,Vector3(0.0f,0.0f,0.0f),"min", "Min")),
      max(initData(&max,Vector3(1.0f,1.0f,1.0f),"max", "Max")),
      p0(initData(&p0,Vector3(0.0f,0.0f,0.0f),"p0", "p0")),
      _cellWidth(initData(&_cellWidth, (SReal)0.0, "cellWidth","if > 0 : dimension of each cell in the created grid"))

{
}

void RegularGridTopology::init()
{
    if (_cellWidth.getValue())
    {
        SReal w = _cellWidth.getValue();

        Vec3i grid;
        grid[0]= (int)ceil((max.getValue()[0]-min.getValue()[0]) / w)+1;
        grid[1]= (int)ceil((max.getValue()[1]-min.getValue()[1]) / w)+1;
        grid[2]= (int)ceil((max.getValue()[2]-min.getValue()[2]) / w)+1;
        n.setValue(grid);
        setSize();
        sout << "Grid size: " << n.getValue() << sendl;
    }

    if (computeHexaList.getValue())
    {
        updateHexahedra();
        const SeqHexahedra seq_hexa= this->getHexahedra();
        sout<<"Init: Number of Hexadredra ="<<seq_hexa.size()<<sendl;
    }

    if (computeQuadList.getValue())
    {
        //updateQuads();
        const SeqQuads seq_quads= this->getQuads();
        sout<<"Init: Number of Quads ="<<seq_quads.size()<<sendl;
    }

    if (computeEdgeList.getValue())
    {
        //updateEdges();
        const SeqLines seq_l=this->getLines();
        sout<<"Init: Number of Lines ="<<seq_l.size()<<sendl;
    }

    if (computePointList.getValue())
    {
        int nbPoints= this->getNbPoints();
        // put the result in seqPoints
        SeqPoints& seq_P= *(seqPoints.beginEdit());
        seq_P.resize(nbPoints);

        for (int i=0; i<nbPoints; i++)
        {
            seq_P[i] = this->getPoint(i);
        }

        seqPoints.endEdit();
    }

    //    MeshTopology::init();

    Inherit1::init();
}

void RegularGridTopology::setPos(BoundingBox b)
{
    Vector3 m=b.minBBox(), M=b.maxBBox();
    setPos(m[0],M[0],m[1],M[1],m[2],M[2]);
}

void RegularGridTopology::setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax)
{
    SReal p0x=xmin, p0y=ymin, p0z=zmin;

    if (n.getValue()[0]>1)
        setDx(Vector3((xmax-xmin)/(n.getValue()[0]-1),0,0));
    else
    {
        setDx(Vector3(xmax-xmin,0,0));
        p0x = (xmax+xmin)/2;
    }

    if (n.getValue()[1]>1)
        setDy(Vector3(0,(ymax-ymin)/(n.getValue()[1]-1),0));
    else
    {
        setDy(Vector3(0,ymax-ymin,0));
        p0y = (ymax+ymin)/2;
    }

    if (n.getValue()[2]>1)
        setDz(Vector3(0,0,(zmax-zmin)/(n.getValue()[2]-1)));
    else
    {
        setDz(Vector3(0,0,zmax-zmin));
        //p0z = (zmax+zmin)/2;
        p0z = zmin;
    }

    min.setValue(Vector3(xmin,ymin,zmin));
    max.setValue(Vector3(xmax,ymax,zmax));
    if (!p0.isSet())
    {
        setP0(Vector3(p0x,p0y,p0z));
    }
}

unsigned RegularGridTopology::getIndex( int i, int j, int k ) const
{
    return n.getValue()[0]* ( n.getValue()[1]*k + j ) + i;
}


Vector3 RegularGridTopology::getPoint(int i) const
{

    int x = i%n.getValue()[0]; i/=n.getValue()[0];
    int y = i%n.getValue()[1]; i/=n.getValue()[1];
    int z = i;

    return getPoint(x,y,z);
}

Vector3 RegularGridTopology::getPoint(int x, int y, int z) const
{
    return p0.getValue()+dx*x+dy*y+dz*z;
}

/// return the cube containing the given point (or -1 if not found).
int RegularGridTopology::findCube(const Vector3& pos)
{
    if (n.getValue()[0]<2 || n.getValue()[1]<2 || n.getValue()[2]<2)
        return -1;
    Vector3 p = pos-p0.getValue();
    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (   (unsigned)ix <= (unsigned)n.getValue()[0]-2
            && (unsigned)iy <= (unsigned)n.getValue()[1]-2
            && (unsigned)iz <= (unsigned)n.getValue()[2]-2 )
    {
        return cube(ix,iy,iz);
    }
    else
    {
        return -1;
    }
}

/// return the nearest cube (or -1 if not found).
int RegularGridTopology::findNearestCube(const Vector3& pos)
{
    if (n.getValue()[0]<2 || n.getValue()[1]<2 || n.getValue()[2]<2) return -1;
    Vector3 p = pos-p0.getValue();
    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (ix<0) ix=0; else if (ix>n.getValue()[0]-2) ix=n.getValue()[0]-2;
    if (iy<0) iy=0; else if (iy>n.getValue()[1]-2) iy=n.getValue()[1]-2;
    if (iz<0) iz=0; else if (iz>n.getValue()[2]-2) iz=n.getValue()[2]-2;
    return cube(ix,iy,iz);
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int RegularGridTopology::findCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    if (n.getValue()[0]<2 || n.getValue()[1]<2 || n.getValue()[2]<2) return -1;
    Vector3 p = pos-p0.getValue();

    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;

    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if ((unsigned)ix<=(unsigned)n.getValue()[0]-2 && (unsigned)iy<=(unsigned)n.getValue()[1]-2 && (unsigned)iz<=(unsigned)n.getValue()[2]-2)
    {
        fx = x-ix;
        fy = y-iy;
        fz = z-iz;
        return cube(ix,iy,iz);
    }
    else
    {
        return -1;
    }
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int RegularGridTopology::findNearestCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    if (n.getValue()[0]<2 || n.getValue()[1]<2 || n.getValue()[2]<2) return -1;
    Vector3 p = pos-p0.getValue();
    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (ix<0) ix=0; else if (ix>n.getValue()[0]-2) ix=n.getValue()[0]-2;
    if (iy<0) iy=0; else if (iy>n.getValue()[1]-2) iy=n.getValue()[1]-2;
    if (iz<0) iz=0; else if (iz>n.getValue()[2]-2) iz=n.getValue()[2]-2;
    fx = x-ix;
    fy = y-iy;
    fz = z-iz;
    return cube(ix,iy,iz);
}


unsigned RegularGridTopology::getCubeIndex( int i, int j, int k ) const
{
    return (n.getValue()[0]-1)* ( (n.getValue()[1]-1)*k + j ) + i;
}

Vector3 RegularGridTopology::getCubeCoordinate(int i) const
{
    Vector3 result;
    result[0] = (SReal)(i%(n.getValue()[0]-1)); i/=(n.getValue()[0]-1);
    result[1] = (SReal)(i%(n.getValue()[1]-1)); i/=(n.getValue()[1]-1);
    result[2] = (SReal)i;
    return result;
}

void RegularGridTopology::createTexCoords()
{
#ifndef NDEBUG
    std::cout << "createTexCoords" << std::endl;
#endif
    unsigned int nPts = this->getNbPoints();
    const Vec3i& _n = n.getValue();

    if ( (_n[0] == 1 && _n[1] == 1) || (_n[0] == 1 && _n[2] == 1) || (_n[1] == 1 && _n[2] == 1))
    {
        std::cerr << "Error: can't create Texture coordinates as at least 2 dimensions of the grid are null."  << std::endl;
        return;
    }

#ifndef NDEBUG
    std::cout << "nbP: " << nPts << std::endl;
#endif
    helper::WriteAccessor< Data< vector<Vector2> > > _texCoords = this->seqUVs;
    _texCoords.resize(nPts);

    // check if flat grid
    Vec3ui axes;

    if (_n[0] == 1)
        axes = Vec3i(1, 2, 0);
    else if (_n[1] == 1)
        axes = Vec3i(0, 2, 1);
    else
        axes = Vec3i(0, 1, 2);

    SReal Uscale = 1/(SReal)(_n[ axes[0] ]-1);
    SReal Vscale = 1/(SReal)(_n[ axes[1] ]-1);

    for (int n1 = 0; n1 < _n[ axes[1] ]; ++n1)
    {
        for (int n0 = 0; n0 < _n[ axes[0] ]; ++n0)
        {
            unsigned int pt1 = n0 + _n[ axes[0] ] * n1;
            unsigned int pt2 = n0 + _n[ axes[0] ] * (n1 + _n[ axes[1] ] * (_n[ axes[2] ]-1));
#ifndef NDEBUG
            std::cout << "pt1: " << pt1 << std::endl;
            std::cout << "pt2: " << pt2 << std::endl;
#endif
            _texCoords[pt1] = Vector2(n0*Uscale, n1*Vscale);
            _texCoords[pt2] = Vector2(1- n0*Uscale, 1 - n1*Vscale);
        }
    }

    if (_n[ axes[2] ] != 0)
    {
        Uscale = 1/(SReal)(_n[ axes[0] ]-1);
        Vscale = 1/(SReal)(_n[ axes[2] ]-1);

        for (int n2 = 1; n2 < _n[ axes[2] ]-1; ++n2)
        {
            for (int n0 = 1; n0 < _n[ axes[0] ]-1; ++n0)
            {
                unsigned int pt1 = n0 + _n[ axes[0] ] * n2;
                unsigned int pt2 = n0 + _n[ axes[0] ] * (n2 + _n[ axes[2] ] * (_n[ axes[1] ]-1));
                _texCoords[pt1] = Vector2(n0*Uscale, n2*Vscale);
                _texCoords[pt2] = Vector2(1- n0*Uscale, 1 - n2*Vscale);
            }
        }


        Uscale = 1/(SReal)(_n[ axes[2] ]-1);
        Vscale = 1/(SReal)(_n[ axes[1] ]-1);

        for (int n1 = 1; n1 < _n[ axes[1] ]-1; ++n1)
        {
            for (int n2 = 1; n2 < _n[ axes[2] ]-1; ++n2)
            {
                unsigned int pt1 = n2 + _n[ axes[2] ] * n1;
                unsigned int pt2 = n2 + _n[ axes[2] ] * (n1 + _n[ axes[1] ] * (_n[ axes[0] ])-1);
                _texCoords[pt1] = Vector2(n2*Uscale, n1*Vscale);
                _texCoords[pt2] = Vector2(1- n2*Uscale, 1 - n1*Vscale);
            }
        }
    }
}


} // namespace topology

} // namespace component

} // namespace sofa

