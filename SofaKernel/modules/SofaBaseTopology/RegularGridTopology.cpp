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
using helper::vector;

/// To avoid duplicating the code in the different variants of the constructor
/// this object is using the delegating constructor feature of c++ x11.
/// The following constructor is "chained" by the other constructors to
/// defined only one the member initialization.
RegularGridTopology::RegularGridTopology(const Vec3i& dimXYZ)
    : GridTopology(dimXYZ)
    , d_min(initData(&d_min,Vector3(0.0f,0.0f,0.0f),"min", "Min end of the diagonal"))
    , d_max(initData(&d_max,Vector3(1.0f,1.0f,1.0f),"max", "Max end of the diagonal"))
    , d_p0(initData(&d_p0,Vector3(0.0f,0.0f,0.0f),"p0", "Offset all the grid points"))
    , d_cellWidth(initData(&d_cellWidth, (SReal)0.0, "cellWidth","if > 0 : dimension of each cell in the created grid. Otherwise, the cell size is computed based on min, max, and resolution n."))
{
}

RegularGridTopology::RegularGridTopology()
    : RegularGridTopology(Vec3i(2,2,2))
{
}

RegularGridTopology::RegularGridTopology(int nx, int ny, int nz)
    : RegularGridTopology(Vec3i(nx,ny,nz))
{
}

RegularGridTopology::RegularGridTopology(const Vec3i& n, BoundingBox b)
    : RegularGridTopology(n)
{
    setPos(b);
}


void RegularGridTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    float scale=1.0f;
    if (arg->getAttribute("scale")!=NULL)
    {
        scale = arg->getAttributeAsFloat("scale", 1.0);
    }

    this->GridTopology::parse(arg);
    if (arg->getAttribute("xmin") != NULL &&
        arg->getAttribute("ymin") != NULL &&
        arg->getAttribute("zmin") != NULL &&
        arg->getAttribute("xmax") != NULL &&
        arg->getAttribute("ymax") != NULL &&
        arg->getAttribute("zmax") != NULL )
    {
        float xmin = arg->getAttributeAsFloat("xmin",0);
        float ymin = arg->getAttributeAsFloat("ymin",0);
        float zmin = arg->getAttributeAsFloat("zmin",0);
        float xmax = arg->getAttributeAsFloat("xmax",1);
        float ymax = arg->getAttributeAsFloat("ymax",1);
        float zmax = arg->getAttributeAsFloat("zmax",1);
        d_min.setValue(Vector3((SReal)xmin*scale, (SReal)ymin*scale, (SReal)zmin*scale));
        d_max.setValue(Vector3((SReal)xmax*scale, (SReal)ymax*scale, (SReal)zmax*scale));
    }
    this->setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);

}

void RegularGridTopology::init()
{
    if (d_cellWidth.getValue())
    {
        SReal w = d_cellWidth.getValue();

        Vec3i grid;
        grid[0]= (int)ceil((d_max.getValue()[0]-d_min.getValue()[0]) / w)+1;
        grid[1]= (int)ceil((d_max.getValue()[1]-d_min.getValue()[1]) / w)+1;
        grid[2]= (int)ceil((d_max.getValue()[2]-d_min.getValue()[2]) / w)+1;
        d_n.setValue(grid);
        setNbGridPoints();
    }

    Inherit1::init();
}

void RegularGridTopology::reinit()
{
    setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);

    Inherit1::reinit();
}

void RegularGridTopology::changeGridResolutionPostProcess()
{
    setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);
}

void RegularGridTopology::setPos(BoundingBox b)
{
    Vector3 m=b.minBBox(), M=b.maxBBox();
    setPos(m[0],M[0],m[1],M[1],m[2],M[2]);
}

void RegularGridTopology::setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax)
{
    SReal p0x=xmin, p0y=ymin, p0z=zmin;

    if (d_n.getValue()[0]>1)
        setDx(Vector3((xmax-xmin)/(d_n.getValue()[0]-1),0,0));
    else
    {
        setDx(Vector3(xmax-xmin,0,0));
        p0x = (xmax+xmin)/2;
    }

    if (d_n.getValue()[1]>1)
        setDy(Vector3(0,(ymax-ymin)/(d_n.getValue()[1]-1),0));
    else
    {
        setDy(Vector3(0,ymax-ymin,0));
        p0y = (ymax+ymin)/2;
    }

    if (d_n.getValue()[2]>1)
        setDz(Vector3(0,0,(zmax-zmin)/(d_n.getValue()[2]-1)));
    else
    {
        setDz(Vector3(0,0,zmax-zmin));
        //p0z = (zmax+zmin)/2;
        p0z = zmin;
    }

    d_min.setValue(Vector3(xmin,ymin,zmin));
    d_max.setValue(Vector3(xmax,ymax,zmax));
    if (!d_p0.isSet())
    {
        setP0(Vector3(p0x,p0y,p0z));
    }
}

Vector3 RegularGridTopology::getPointInGrid(int i, int j, int k) const
{
    return d_p0.getValue()+dx*i+dy*j+dz*k;
}


/// return the cube containing the given point (or -1 if not found).
int RegularGridTopology::findCube(const Vector3& pos)
{
    if (d_n.getValue()[0]<2 || d_n.getValue()[1]<2 || d_n.getValue()[2]<2)
        return -1;
    Vector3 p = pos-d_p0.getValue();
    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (   (unsigned)ix <= (unsigned)d_n.getValue()[0]-2
            && (unsigned)iy <= (unsigned)d_n.getValue()[1]-2
            && (unsigned)iz <= (unsigned)d_n.getValue()[2]-2 )
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
    if (d_n.getValue()[0]<2 || d_n.getValue()[1]<2 || d_n.getValue()[2]<2) return -1;
    Vector3 p = pos-d_p0.getValue();
    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (ix<0) ix=0; else if (ix>d_n.getValue()[0]-2) ix=d_n.getValue()[0]-2;
    if (iy<0) iy=0; else if (iy>d_n.getValue()[1]-2) iy=d_n.getValue()[1]-2;
    if (iz<0) iz=0; else if (iz>d_n.getValue()[2]-2) iz=d_n.getValue()[2]-2;
    return cube(ix,iy,iz);
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int RegularGridTopology::findCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    if (d_n.getValue()[0]<2 || d_n.getValue()[1]<2 || d_n.getValue()[2]<2) return -1;
    Vector3 p = pos-d_p0.getValue();

    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;

    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if ((unsigned)ix<=(unsigned)d_n.getValue()[0]-2 && (unsigned)iy<=(unsigned)d_n.getValue()[1]-2 && (unsigned)iz<=(unsigned)d_n.getValue()[2]-2)
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
    if (d_n.getValue()[0]<2 || d_n.getValue()[1]<2 || d_n.getValue()[2]<2) return -1;
    Vector3 p = pos-d_p0.getValue();
    SReal x = p*dx*inv_dx2;
    SReal y = p*dy*inv_dy2;
    SReal z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (ix<0) ix=0; else if (ix>d_n.getValue()[0]-2) ix=d_n.getValue()[0]-2;
    if (iy<0) iy=0; else if (iy>d_n.getValue()[1]-2) iy=d_n.getValue()[1]-2;
    if (iz<0) iz=0; else if (iz>d_n.getValue()[2]-2) iz=d_n.getValue()[2]-2;
    fx = x-ix;
    fy = y-iy;
    fz = z-iz;
    return cube(ix,iy,iz);
}


unsigned RegularGridTopology::getCubeIndex( int i, int j, int k ) const
{
    return (d_n.getValue()[0]-1)* ( (d_n.getValue()[1]-1)*k + j ) + i;
}

Vector3 RegularGridTopology::getCubeCoordinate(int i) const
{
    Vector3 result;
    result[0] = (SReal)(i%(d_n.getValue()[0]-1)); i/=(d_n.getValue()[0]-1);
    result[1] = (SReal)(i%(d_n.getValue()[1]-1)); i/=(d_n.getValue()[1]-1);
    result[2] = (SReal)i;
    return result;
}

void RegularGridTopology::createTexCoords()
{
    unsigned int nPts = this->getNbPoints();
    const Vec3i& _n = d_n.getValue();

    if ( (_n[0] == 1 && _n[1] == 1) || (_n[0] == 1 && _n[2] == 1) || (_n[1] == 1 && _n[2] == 1))
    {
        msg_warning() << "Can't create Texture coordinates as at least 2 dimensions of the grid are null." ;
        return;
    }

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


SOFA_DECL_CLASS(RegularGridTopology)

int RegularGridTopologyClass = core::RegisterObject("Regular grid in 3D")
        .addAlias("RegularGrid")
        .add< RegularGridTopology >()
        ;


} // namespace topology

} // namespace component

} // namespace sofa

