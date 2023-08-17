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
#include <sofa/component/topology/container/grid/RegularGridTopology.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::grid
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using type::vector;

/// To avoid duplicating the code in the different variants of the constructor
/// this object is using the delegating constructor feature of c++ x11.
/// The following constructor is "chained" by the other constructors to
/// defined only one the member initialization.
RegularGridTopology::RegularGridTopology(const type::Vec3i& dimXYZ)
    : GridTopology(dimXYZ)
    , d_min(initData(&d_min,type::Vec3(0_sreal, 0_sreal, 0_sreal),"min", "Min end of the diagonal"))
    , d_max(initData(&d_max,type::Vec3(1_sreal, 1_sreal, 1_sreal),"max", "Max end of the diagonal"))
    , d_p0(initData(&d_p0,type::Vec3(0_sreal, 0_sreal, 0_sreal),"p0", "Offset all the grid points"))
    , d_cellWidth(initData(&d_cellWidth, 0_sreal, "cellWidth","if > 0 : dimension of each cell in the created grid. Otherwise, the cell size is computed based on min, max, and resolution n."))
{
}

RegularGridTopology::RegularGridTopology()
    : RegularGridTopology(type::Vec3i(2,2,2))
{
}

RegularGridTopology::RegularGridTopology(int nx, int ny, int nz)
    : RegularGridTopology(type::Vec3i(nx,ny,nz))
{
}

RegularGridTopology::RegularGridTopology(const type::Vec3i& n, type::BoundingBox b)
    : RegularGridTopology(n)
{
    setPos(b);
}


void RegularGridTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    float scale=1.0f;
    if (arg->getAttribute("scale")!=nullptr)
    {
        scale = arg->getAttributeAsFloat("scale", 1.0);
    }

    this->GridTopology::parse(arg);
    if (arg->getAttribute("xmin") != nullptr &&
        arg->getAttribute("ymin") != nullptr &&
        arg->getAttribute("zmin") != nullptr &&
        arg->getAttribute("xmax") != nullptr &&
        arg->getAttribute("ymax") != nullptr &&
        arg->getAttribute("zmax") != nullptr )
    {
        const float xmin = arg->getAttributeAsFloat("xmin",0);
        const float ymin = arg->getAttributeAsFloat("ymin",0);
        const float zmin = arg->getAttributeAsFloat("zmin",0);
        const float xmax = arg->getAttributeAsFloat("xmax",1);
        const float ymax = arg->getAttributeAsFloat("ymax",1);
        const float zmax = arg->getAttributeAsFloat("zmax",1);
        d_min.setValue(type::Vec3((SReal)xmin*scale, (SReal)ymin*scale, (SReal)zmin*scale));
        d_max.setValue(type::Vec3((SReal)xmax*scale, (SReal)ymax*scale, (SReal)zmax*scale));
    }
    this->setPos(d_min.getValue()[0],d_max.getValue()[0],d_min.getValue()[1],d_max.getValue()[1],d_min.getValue()[2],d_max.getValue()[2]);

}

void RegularGridTopology::init()
{
    if (d_cellWidth.getValue())
    {
        const SReal w = d_cellWidth.getValue();

        type::Vec3i grid;
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

void RegularGridTopology::setPos(type::BoundingBox b)
{
    type::Vec3 m=b.minBBox(), M=b.maxBBox();
    setPos(m[0],M[0],m[1],M[1],m[2],M[2]);
}

void RegularGridTopology::setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax)
{
    SReal p0x=xmin, p0y=ymin, p0z=zmin;
    const type::Vec3i _n = d_n.getValue() - type::Vec3i(1,1,1);

    if (_n[0] > 0)
        setDx(type::Vec3((xmax - xmin) / _n[0], 0_sreal, 0_sreal));
    else
    {
        setDx(type::Vec3(xmax-xmin, 0_sreal, 0_sreal));
        p0x = (xmax+xmin)/2;
    }

    if (_n[1] > 0)
        setDy(type::Vec3(0_sreal, (ymax - ymin) / _n[1], 0_sreal));
    else
    {
        setDy(type::Vec3(0_sreal, ymax-ymin, 0_sreal));
        p0y = (ymax+ymin)/2;
    }

    if (_n[2] > 0)
        setDz(type::Vec3(0_sreal, 0_sreal, (zmax - zmin) / _n[2]));
    else
    {
        setDz(type::Vec3(0_sreal, 0_sreal, zmax-zmin));
        //p0z = (zmax+zmin)/2;
        p0z = zmin;
    }

    d_min.setValue(type::Vec3(xmin,ymin,zmin));
    d_max.setValue(type::Vec3(xmax,ymax,zmax));
    if (!d_p0.isSet())
    {
        setP0(type::Vec3(p0x,p0y,p0z));
    }
}

sofa::type::Vec3 RegularGridTopology::getPointInGrid(int i, int j, int k) const
{
    return d_p0.getValue()+dx*i+dy*j+dz*k;
}

RegularGridTopology::Index RegularGridTopology::findPoint(const type::Vec3& position) const
{
    const type::Vec3 p0 = d_p0.getValue();
    const type::Vec3i   n  = d_n.getValue();
    const type::Vec3 d (dx[0], dy[1], dz[2]);

    // Get the position relative to the corner of the grid
    const type::Vec3 p = position-p0;

    // Get the index of the closest node by rounding the number of cells in the x, y and z direction
    // from the corner of the grid to the queried point
    const int ix = int(round(p[0]/d[0]));
    const int iy = int(round(p[1]/d[1]));
    const int iz = int(round(p[2]/d[2]));

    // Make sure the node lies inside the boundaries of the grid
    if (ix < 0 || iy < 0 || iz < 0)
        return InvalidID;

    if (ix > (n[0] - 1) || iy > (n[1] - 1) || iz > (n[2] - 1))
        return InvalidID;

    // Return the node index
    return getIndex(ix, iy, iz);
}

RegularGridTopology::Index RegularGridTopology::findPoint(const type::Vec3& position, const SReal epsilon) const
{
    const type::Vec3 p0 = d_p0.getValue();
    const type::Vec3i   n  = d_n.getValue();
    const type::Vec3 d (dx[0], dy[1], dz[2]);

    // Get the position relative to the corner of the grid
    const type::Vec3 p = position-p0;

    // Get the index of the closest node by rounding the number of cells in the x, y and z direction
    // from the corner of the grid to the queried point
    const int ix = int(round(p[0]/d[0]));
    const int iy = int(round(p[1]/d[1]));
    const int iz = int(round(p[2]/d[2]));

    // Make sure the node lies inside the boundaries of the grid
    if (ix < 0 || iy < 0 || iz < 0)
        return InvalidID;

    if (ix > n[0] || iy > n[1]|| iz > n[2])
        return InvalidID;

    // Get the node index
    const auto node_index = getIndex(ix, iy, iz);

    // Make sure the node lies inside a sphere of radius (d * epsilon) centered on the queried position
    const auto node_position = type::Vec3(ix*d[0], iy*d[1], iz*d[2]);
    const auto m = epsilon*d[0]; // allowed margin
    const auto e = p-node_position; // vector between the node and the queried position

    if (e[0]*e[0] + e[1]*e[1]+ e[2]*e[2] > m*m)
        return InvalidID;

    return node_index;
}

/// return the cube containing the given point (or -1 if not found).
RegularGridTopology::Index RegularGridTopology::findCube(const type::Vec3& pos)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2)
        return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();
    const SReal x = p*dx*inv_dx2;
    const SReal y = p*dy*inv_dy2;
    const SReal z = p*dz*inv_dz2;
    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));
    if (   (unsigned)ix <= (unsigned)n[0]-2
            && (unsigned)iy <= (unsigned)n[1]-2
            && (unsigned)iz <= (unsigned)n[2]-2 )
    {
        return cube(ix,iy,iz);
    }
    else
    {
        return InvalidID;
    }
}

/// return the nearest cube (or -1 if not found).
RegularGridTopology::Index RegularGridTopology::findNearestCube(const type::Vec3& pos)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2) return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();
    const SReal x = p*dx*inv_dx2;
    const SReal y = p*dy*inv_dy2;
    const SReal z = p*dz*inv_dz2;
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    int iz = int(std::floor(z));
    if (ix<0) ix=0; else if (ix>n[0]-2) ix=n[0]-2;
    if (iy<0) iy=0; else if (iy>n[1]-2) iy=n[1]-2;
    if (iz<0) iz=0; else if (iz>n[2]-2) iz=n[2]-2;
    return cube(ix,iy,iz);
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
RegularGridTopology::Index RegularGridTopology::findCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2) return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();

    const SReal x = p*dx*inv_dx2;
    const SReal y = p*dy*inv_dy2;
    const SReal z = p*dz*inv_dz2;

    const int ix = int(std::floor(x));
    const int iy = int(std::floor(y));
    const int iz = int(std::floor(z));

    if ((unsigned)ix<=(unsigned)n[0]-2 && (unsigned)iy<=(unsigned)n[1]-2 && (unsigned)iz<=(unsigned)n[2]-2)
    {
        fx = x-ix;
        fy = y-iy;
        fz = z-iz;
        return cube(ix,iy,iz);
    }
    else
    {
        return InvalidID;
    }
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
RegularGridTopology::Index RegularGridTopology::findNearestCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    const auto n = d_n.getValue();
    if (n[0]<2 || n[1]<2 || n[2]<2) return InvalidID;
    const type::Vec3 p = pos-d_p0.getValue();
    const SReal x = p*dx*inv_dx2;
    const SReal y = p*dy*inv_dy2;
    const SReal z = p*dz*inv_dz2;
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    int iz = int(std::floor(z));
    if (ix<0) ix=0; else if (ix>n[0]-2) ix=n[0]-2;
    if (iy<0) iy=0; else if (iy>n[1]-2) iy=n[1]-2;
    if (iz<0) iz=0; else if (iz>n[2]-2) iz=n[2]-2;
    fx = x-ix;
    fy = y-iy;
    fz = z-iz;
    return cube(ix,iy,iz);
}


RegularGridTopology::Index RegularGridTopology::getCubeIndex( int i, int j, int k ) const
{
    const auto n = d_n.getValue();
    return (n[0]-1)* ( (n[1]-1)*k + j ) + i;
}

sofa::type::Vec3 RegularGridTopology::getCubeCoordinate(RegularGridTopology::Index i) const
{
    const auto n = d_n.getValue();
    type::Vec3 result;
    result[0] = (SReal)(i%(n[0]-1)); i/=(n[0]-1);
    result[1] = (SReal)(i%(n[1]-1)); i/=(n[1]-1);
    result[2] = (SReal)i;
    return result;
}

void RegularGridTopology::createTexCoords()
{
    const std::size_t nPts = this->getNbPoints();
    const type::Vec3i& _n = d_n.getValue();

    if ( (_n[0] == 1 && _n[1] == 1) || (_n[0] == 1 && _n[2] == 1) || (_n[1] == 1 && _n[2] == 1))
    {
        msg_warning() << "Can't create Texture coordinates as at least 2 dimensions of the grid are null." ;
        return;
    }

    auto _texCoords = sofa::helper::getWriteAccessor(this->seqUVs);
    _texCoords.resize(nPts);

    // check if flat grid
    type::Vec3u axes;

    if (_n[0] == 1)
        axes = type::Vec3u(1u, 2u, 0u);
    else if (_n[1] == 1)
        axes = type::Vec3u(0u, 2u, 1u);
    else
        axes = type::Vec3u(0u, 1u, 2u);

    SReal Uscale = 1/(SReal)(_n[ axes[0] ]-1);
    SReal Vscale = 1/(SReal)(_n[ axes[1] ]-1);

    for (int n1 = 0; n1 < _n[ axes[1] ]; ++n1)
    {
        for (int n0 = 0; n0 < _n[ axes[0] ]; ++n0)
        {
            const unsigned int pt1 = n0 + _n[ axes[0] ] * n1;
            const unsigned int pt2 = n0 + _n[ axes[0] ] * (n1 + _n[ axes[1] ] * (_n[ axes[2] ]-1));
            _texCoords[pt1] = Vec2(n0*Uscale, n1*Vscale);
            _texCoords[pt2] = Vec2(1- n0*Uscale, 1 - n1*Vscale);
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
                const unsigned int pt1 = n0 + _n[ axes[0] ] * n2;
                const unsigned int pt2 = n0 + _n[ axes[0] ] * (n2 + _n[ axes[2] ] * (_n[ axes[1] ]-1));
                _texCoords[pt1] = Vec2(n0*Uscale, n2*Vscale);
                _texCoords[pt2] = Vec2(1- n0*Uscale, 1 - n2*Vscale);
            }
        }


        Uscale = 1/(SReal)(_n[ axes[2] ]-1);
        Vscale = 1/(SReal)(_n[ axes[1] ]-1);

        for (int n1 = 1; n1 < _n[ axes[1] ]-1; ++n1)
        {
            for (int n2 = 1; n2 < _n[ axes[2] ]-1; ++n2)
            {
                const unsigned int pt1 = n2 + _n[ axes[2] ] * n1;
                const unsigned int pt2 = n2 + _n[ axes[2] ] * (n1 + _n[ axes[1] ] * (_n[ axes[0] ])-1);
                _texCoords[pt1] = Vec2(n2*Uscale, n1*Vscale);
                _texCoords[pt2] = Vec2(1- n2*Uscale, 1 - n1*Vscale);
            }
        }
    }
}


int RegularGridTopologyClass = core::RegisterObject("Regular grid in 3D")
        .addAlias("RegularGrid")
        .add< RegularGridTopology >()
        ;


} //namespace sofa::component::topology::container::grid
