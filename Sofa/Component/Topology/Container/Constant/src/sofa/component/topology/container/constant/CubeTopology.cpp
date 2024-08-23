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
#include <sofa/component/topology/container/constant/CubeTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::constant
{

using namespace sofa::type;
using namespace sofa::defaulttype;

void CubeTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->MeshTopology::parse(arg);
    float scale=1.0f;
    if (arg->getAttribute("scale")!=nullptr)
    {
        scale = arg->getAttributeAsFloat("scale",1.0);
    }
    this->setSize();
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
        d_min.setValue(Vec3((SReal)(xmin * scale), (SReal)(ymin * scale), (SReal)(zmin * scale)));
        d_max.setValue(Vec3((SReal)(xmax * scale), (SReal)(ymax * scale), (SReal)(zmax * scale)));
    }
    this->setPos(d_min.getValue()[0], d_max.getValue()[0], d_min.getValue()[1], d_max.getValue()[1], d_min.getValue()[2], d_max.getValue()[2]);
}

int CubeTopologyClass = core::RegisterObject("Surface of a cube in 3D")
        .add< CubeTopology >()
        ;

CubeTopology::CubeTopology(int _nx, int _ny, int _nz)
    : d_nx(initData(&d_nx, _nx, "nx", "x grid resolution"))
    , d_ny(initData(&d_ny, _ny, "ny", "y grid resolution"))
    , d_nz(initData(&d_nz, _nz, "nz", "z grid resolution"))
    , d_internalPoints(initData(&d_internalPoints, false, "internalPoints", "include internal points (allow a one-to-one mapping between points from RegularGridTopology and CubeTopology)"))
    , d_splitNormals(initData(&d_splitNormals, false, "splitNormals", "split corner points to have planar normals"))
    , d_min(initData(&d_min, Vec3(0.0_sreal, 0.0_sreal, 0.0_sreal), "min", "Min"))
    , d_max(initData(&d_max, Vec3(1.0_sreal, 1.0_sreal, 1.0_sreal), "max", "Max"))
{
    setSize();
    nx.setOriginalData(&d_nx);
    ny.setOriginalData(&d_ny);
    nz.setOriginalData(&d_nz);
    internalPoints.setOriginalData(&d_internalPoints);
    splitNormals.setOriginalData(&d_splitNormals);
    min.setOriginalData(&d_min);
    max.setOriginalData(&d_max);
}

CubeTopology::CubeTopology()
    : d_nx(initData(&d_nx, 0, "nx", "x grid resolution")), d_ny(initData(&d_ny, 0, "ny", "y grid resolution")), d_nz(initData(&d_nz, 0, "nz", "z grid resolution"))
    , d_internalPoints(initData(&d_internalPoints, false, "internalPoints", "include internal points (allow a one-to-one mapping between points from RegularGridTopology and CubeTopology)"))
    , d_splitNormals(initData(&d_splitNormals, false, "splitNormals", "split corner points to have planar normals"))
    , d_min(initData(&d_min, Vec3(0.0_sreal, 0.0_sreal, 0.0_sreal), "min", "Min"))
    , d_max(initData(&d_max, Vec3(1.0_sreal, 1.0_sreal, 1.0_sreal), "max", "Max"))
{
    nx.setOriginalData(&d_nx);
    ny.setOriginalData(&d_ny);
    nz.setOriginalData(&d_nz);
    internalPoints.setOriginalData(&d_internalPoints);
    splitNormals.setOriginalData(&d_splitNormals);
    min.setOriginalData(&d_min);
    max.setOriginalData(&d_max);
}

void CubeTopology::setSize(int nx, int ny, int nz)
{
    if (nx == this->d_nx.getValue() && ny == this->d_ny.getValue() && nz == this->d_nz.getValue())
        return;
    this->d_nx.setValue(nx);
    this->d_ny.setValue(ny);
    this->d_nz.setValue(nz);
    setSize();
}

void CubeTopology::init()
{
    this->MeshTopology::init();
    setSize();
}

void CubeTopology::reinit()
{
    this->MeshTopology::reinit();
    setSize();
}


void CubeTopology::setSize()
{
    const int nx = this->d_nx.getValue();
    const int ny = this->d_ny.getValue();
    const int nz = this->d_nz.getValue();
    if (d_splitNormals.getValue())
        this->nbPoints = nx*ny*(nz>1?2:1) + ny*nz*(nx>1?2:1) + nz*nx*(ny>1?2:1);
    else if (!d_internalPoints.getValue() && (nx > 1 && ny > 1 && nz > 1))
        this->nbPoints = nx*ny*2 + (nz-2)*(2*nx+2*ny-4);
    else
        this->nbPoints = nx*ny*nz;

    invalidate();

    // FF: add the following functions which seem to be missing, but I am not sure…
    updatePoints();
    updateEdges();
    updateQuads();
}

int CubeTopology::point(int x, int y, int z, Plane p) const
{
    const int nx = this->d_nx.getValue();
    const int ny = this->d_ny.getValue();
    const int nz = this->d_nz.getValue();
    if (d_splitNormals.getValue())
    {
        if (p == PLANE_UNKNOWN)
        {
            if (x==0) p = PLANE_X0;
            else if (x==nx-1) p = PLANE_X1;
            else if (y==0) p = PLANE_Y0;
            else if (y==ny-1) p = PLANE_Y1;
            else if (z==0) p = PLANE_Z0;
            else if (z==nz-1) p = PLANE_Z1;
        }
        int i = 0;
        switch (p)
        {
        case PLANE_X0: i =                               y+ny*z; break;
        case PLANE_X1: i =   ny*nz +   nx*nz +   nx*ny + y+ny*z; break;
        case PLANE_Y0: i =   ny*nz                     + x+nx*z; break;
        case PLANE_Y1: i = 2*ny*nz +   nx*nz +   nx*ny + x+nx*z; break;
        case PLANE_Z0: i =   ny*nz +   nx*nz           + x+nx*y; break;
        case PLANE_Z1: i = 2*ny*nz + 2*nx*nz +   nx*ny + x+nx*y; break;
        case PLANE_UNKNOWN: break;
        }
        return i;
    }
    else if (!d_internalPoints.getValue() && (nx > 1 && ny > 1 && nz > 1))
    {
        if (z==0)
            return (x+nx*y);
        else if (z==nz-1)
            return (x+nx*(y+ny));
        else
        {
            const int base = nx*ny*2 + (2*nx+2*ny-4)*(z-1);
            if (y==0)
                return base + x;
            else if (y==ny-1)
                return base + x + nx;
            else if (x==0)
                return base + 2*nx + (y-1);
            else
                return base + 2*nx + (ny-2) + (y-1);
        }
    }
    else
        return x+nx*(y+ny*z);
}


void CubeTopology::updatePoints()
{
    SeqPoints& points = *d_seqPoints.beginEdit();
    points.resize(nbPoints) ;
    for(Size i=0;i<nbPoints;i++)
    {
        points[i] = getPoint(i);
    }
    d_seqPoints.endEdit();
}

void CubeTopology::updateEdges()
{
    SeqEdges& edges = *d_seqEdges.beginEdit();
    const int nx = this->d_nx.getValue();
    const int ny = this->d_ny.getValue();
    const int nz = this->d_nz.getValue();
    edges.clear();
    edges.reserve((nx-1)*(2*ny+2*nz-4) + (ny-1)*(2*nx+2*nz-4) + (nz-1)*(2*nx+2*ny-4));
    for (int z=0; z<nz; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++)
            {
                // lines along X
                if (x<nx-1 && (y==0 || y==ny-1 || z==0 || z==nz-1))
                    edges.push_back(Edge(point(x,y,z),point(x+1,y,z)));
                // lines along Y
                if (y<ny-1 && (x==0 || x==nx-1 || z==0 || z==nz-1))
                    edges.push_back(Edge(point(x,y,z),point(x,y+1,z)));
                // lines along Z
                if (z<nz-1 && (x==0 || x==nx-1 || y==0 || y==ny-1))
                    edges.push_back(Edge(point(x,y,z),point(x,y,z+1)));
            }
    d_seqEdges.endEdit();
}

void CubeTopology::updateQuads()
{
    d_seqQuads.beginEdit()->clear();
    const int nx = this->d_nx.getValue();
    const int ny = this->d_ny.getValue();
    const int nz = this->d_nz.getValue();
    d_seqQuads.beginEdit()->reserve((nx - 1) * (ny - 1) * (nz > 1 ? 2 : 1) + (nx - 1) * (nz - 1) * (ny > 1 ? 2 : 1) + (ny - 1) * (nz - 1) * (nx > 1 ? 2 : 1));
    // quads along Z=0 plane
    for (int z=0, y=0; y<ny-1; y++)
        for (int x=0; x<nx-1; x++)
            d_seqQuads.beginEdit()->push_back(Quad(point(x, y, z, PLANE_Z0), point(x, y + 1, z, PLANE_Z0), point(x + 1, y + 1, z, PLANE_Z0), point(x + 1, y, z, PLANE_Z0)));
    // quads along Z=NZ-1 plane
    if (nz > 1)
        for (int z=nz-1, y=0; y<ny-1; y++)
            for (int x=0; x<nx-1; x++)
                d_seqQuads.beginEdit()->push_back(Quad(point(x, y, z, PLANE_Z1), point(x + 1, y, z, PLANE_Z1), point(x + 1, y + 1, z, PLANE_Z1), point(x, y + 1, z, PLANE_Z1)));
    // quads along Y=0 plane
    for (int y=0, z=0; z<nz-1; z++)
        for (int x=0; x<nx-1; x++)
            d_seqQuads.beginEdit()->push_back(Quad(point(x, y, z, PLANE_Y0), point(x + 1, y, z, PLANE_Y0), point(x + 1, y, z + 1, PLANE_Y0), point(x, y, z + 1, PLANE_Y0)));
    // quads along Y=NY-1 plane
    if (ny > 1)
        for (int y=ny-1, z=0; z<nz-1; z++)
            for (int x=0; x<nx-1; x++)
                d_seqQuads.beginEdit()->push_back(Quad(point(x, y, z, PLANE_Y1), point(x, y, z + 1, PLANE_Y1), point(x + 1, y, z + 1, PLANE_Y1), point(x + 1, y, z, PLANE_Y1)));
    // quads along X=0 plane
    for (int x=0, z=0; z<nz-1; z++)
        for (int y=0; y<ny-1; y++)
            d_seqQuads.beginEdit()->push_back(Quad(point(x, y, z, PLANE_X0), point(x, y, z + 1, PLANE_X0), point(x, y + 1, z + 1, PLANE_X0), point(x, y + 1, z, PLANE_X0)));
    // quads along X=NX-1 plane
    if (nx > 1)
        for (int x=nx-1, z=0; z<nz-1; z++)
            for (int y=0; y<ny-1; y++)
                d_seqQuads.beginEdit()->push_back(Quad(point(x, y, z, PLANE_X1), point(x, y + 1, z, PLANE_X1), point(x, y + 1, z + 1, PLANE_X1), point(x, y, z + 1, PLANE_X1)));

    d_seqQuads.endEdit();
}

void CubeTopology::setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax)
{
    setP0(Vec3(xmin,ymin,zmin));
    if (d_nx.getValue() > 1)
        setDx(Vec3((xmax-xmin)/(d_nx.getValue() - 1), 0_sreal, 0_sreal));
    else
        setDx(Vec3(0_sreal,0_sreal,0_sreal));
    if (d_ny.getValue() > 1)
        setDy(Vec3(0_sreal,(ymax-ymin)/(d_ny.getValue() - 1), 0_sreal));
    else
        setDy(Vec3(0_sreal,0_sreal,0_sreal));
    if (d_nz.getValue() > 1)
        setDz(Vec3(0_sreal,0_sreal,(zmax-zmin)/(d_nz.getValue() - 1)));
    else
        setDz(Vec3(0_sreal,0_sreal,0_sreal));
}

Vec3 CubeTopology::getPoint(int i) const
{
    const int nx = this->d_nx.getValue();
    const int ny = this->d_ny.getValue();
    const int nz = this->d_nz.getValue();
    int x,y,z;
    if (d_splitNormals.getValue())
    {
        if (i < ny*nz+nx*nz+nx*ny)
        {
            x = 0;
            y = 0;
            z = 0;
        }
        else
        {
            i -= ny*nz+nx*nz+nx*ny;
            x = nx-1;
            y = ny-1;
            z = nz-1;
        }
        if (i < ny*nz)
        {
            y = i % ny;
            z = i / ny;
        }
        else
        {
            i -= ny*nz;
            if (i < nx*nz)
            {
                x = i % nx;
                z = i / nx;
            }
            else
            {
                i -= nx*nz;
                x = i % nx;
                y = i / nx;
            }
        }
    }
    else if (!d_internalPoints.getValue() && (nx > 1 && ny > 1 && nz > 1))
    {
        const int nxny = nx*ny;
        if (i<nxny)
        {
            x = i%nx; i/=nx;
            y = i;
            z = 0;
        }
        else if (i<2*nxny)
        {
            i -= nxny;
            x = i%nx; i/=nx;
            y = i;
            z = nz-1;
        }
        else
        {
            i -= 2*nxny;
            const int psize = (2*nx+2*(ny-2));
            z = i/psize; i-=z*psize;
            z+=1;
            if (i < nx)
            {
                x = i;
                y = 0;
            }
            else if (i < 2*nx)
            {
                x = i-nx;
                y = ny-1;
            }
            else if (i < 2*nx + (ny-2))
            {
                x = 0;
                y = 1+i-2*nx;
            }
            else
            {
                x = nx-1;
                y = 1+i-2*nx-(ny-2);
            }
        }
    }
    else
    {
        x = i%nx; i/=nx;
        y = i%ny; i/=ny;
        z = i;
    }
    return getPoint(x,y,z);
}

Vec3 CubeTopology::getPoint(int x, int y, int z) const
{
    return p0+dx*x+dy*y+dz*z;
}

} // namespace sofa::component::topology::container::constant
