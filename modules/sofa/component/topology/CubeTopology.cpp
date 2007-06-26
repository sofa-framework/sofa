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
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using std::cout;
using std::endl;

void CubeTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->MeshTopology::parse(arg);
    this->setSize();
    const char* xmin = arg->getAttribute("xmin",arg->getAttribute("min","0"));
    const char* ymin = arg->getAttribute("ymin",arg->getAttribute("min","0"));
    const char* zmin = arg->getAttribute("zmin",arg->getAttribute("min","0"));
    const char* xmax = arg->getAttribute("xmax",arg->getAttribute("max",arg->getAttribute("nx","1")));
    const char* ymax = arg->getAttribute("ymax",arg->getAttribute("max",arg->getAttribute("ny","1")));
    const char* zmax = arg->getAttribute("zmax",arg->getAttribute("max",arg->getAttribute("nz","1")));
    this->setPos(atof(xmin),atof(xmax),atof(ymin),atof(ymax),atof(zmin),atof(zmax));
}

SOFA_DECL_CLASS(CubeTopology)

int CubeTopologyClass = core::RegisterObject("Surface of a cube in 3D")
        .addAlias("Cube")
        .add< CubeTopology >()
        ;

CubeTopology::CubeTopology(int _nx, int _ny, int _nz)
    : nx(dataField(&nx,_nx,"nx","x grid resolution")), ny(dataField(&ny,_ny,"ny","y grid resolution")), nz(dataField(&nz,_nz,"nz","z grid resolution")), internalPoints(dataField(&internalPoints, false, "internalPoints", "include internal points (allow a one-to-one mapping between points from RegularGridTopology and CubeTopology)"))
{
    setSize();
}

CubeTopology::CubeTopology()
    : nx(dataField(&nx,0,"nx","x grid resolution")), ny(dataField(&ny,0,"ny","y grid resolution")), nz(dataField(&nz,0,"nz","z grid resolution"))
{
}

void CubeTopology::setSize(int nx, int ny, int nz)
{
    if (nx == this->nx.getValue() && ny == this->ny.getValue() && nz == this->nz.getValue())
        return;
    this->nx.setValue(nx);
    this->ny.setValue(ny);
    this->nz.setValue(nz);
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
    const int nx = this->nx.getValue();
    const int ny = this->ny.getValue();
    const int nz = this->nz.getValue();
    if (!internalPoints.getValue() && (nx>1 && ny>1 && nz>1))
        this->nbPoints = nx*ny*2 + (nz-2)*(2*nx+2*ny-4);
    else
        this->nbPoints = nx*ny*nz;

    invalidate();
}

int CubeTopology::point(int x, int y, int z) const
{
    const int nx = this->nx.getValue();
    const int ny = this->ny.getValue();
    const int nz = this->nz.getValue();
    if (!internalPoints.getValue() && (nx>1 && ny>1 && nz>1))
    {
        if (z==0)
            return (x+nx*y);
        else if (z==nz-1)
            return (x+nx*(y+ny));
        else
        {
            int base = nx*ny*2 + (2*nx+2*ny-4)*(z-1);
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

void CubeTopology::updateLines()
{
    SeqLines& lines = *seqLines.beginEdit();
    const int nx = this->nx.getValue();
    const int ny = this->ny.getValue();
    const int nz = this->nz.getValue();
    lines.clear();
    lines.reserve((nx-1)*(2*ny+2*nz-4) + (ny-1)*(2*nx+2*nz-4) + (nz-1)*(2*nx+2*ny-4));
    for (int z=0; z<nz; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++)
            {
                // lines along X
                if (x<nx-1 && (y==0 || y==ny-1 || z==0 || z==nz-1))
                    lines.push_back(Line(point(x,y,z),point(x+1,y,z)));
                // lines along Y
                if (y<ny-1 && (x==0 || x==nx-1 || z==0 || z==nz-1))
                    lines.push_back(Line(point(x,y,z),point(x,y+1,z)));
                // lines along Z
                if (z<nz-1 && (x==0 || x==nx-1 || y==0 || y==ny-1))
                    lines.push_back(Line(point(x,y,z),point(x,y,z+1)));
            }
    seqLines.endEdit();
}

void CubeTopology::updateQuads()
{
    seqQuads.clear();
    const int nx = this->nx.getValue();
    const int ny = this->ny.getValue();
    const int nz = this->nz.getValue();
    seqQuads.reserve((nx-1)*(ny-1)*(nz>1?2:1)+(nx-1)*(nz-1)*(ny>1?2:1)+(ny-1)*(nz-1)*(nx>1?2:1));
    // quads along Z=0 plane
    for (int z=0, y=0; y<ny-1; y++)
        for (int x=0; x<nx-1; x++)
            seqQuads.push_back(Quad(point(x,y,z),point(x+1,y,z),point(x+1,y+1,z),point(x,y+1,z)));
    // quads along Z=NZ-1 plane
    if (nz > 1)
        for (int z=nz-1, y=0; y<ny-1; y++)
            for (int x=0; x<nx-1; x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x,y+1,z),point(x+1,y+1,z),point(x+1,y,z)));
    // quads along Y=0 plane
    for (int y=0, z=0; z<nz-1; z++)
        for (int x=0; x<nx-1; x++)
            seqQuads.push_back(Quad(point(x,y,z),point(x+1,y,z),point(x+1,y,z+1),point(x,y,z+1)));
    // quads along Y=NY-1 plane
    if (ny > 1)
        for (int y=ny-1, z=0; z<nz-1; z++)
            for (int x=0; x<nx-1; x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x,y,z+1),point(x+1,y,z+1),point(x+1,y,z)));
    // quads along X=0 plane
    for (int x=0, z=0; z<nz-1; z++)
        for (int y=0; y<ny-1; y++)
            seqQuads.push_back(Quad(point(x,y,z),point(x,y+1,z),point(x,y+1,z+1),point(x,y,z+1)));
    // quads along X=NX-1 plane
    if (nx > 1)
        for (int x=nx-1, z=0; z<nz-1; z++)
            for (int y=0; y<ny-1; y++)
                seqQuads.push_back(Quad(point(x,y,z),point(x,y,z+1),point(x,y+1,z+1),point(x,y+1,z)));
}

void CubeTopology::setPos(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
{
    setP0(Vec3(xmin,ymin,zmin));
    if (nx.getValue()>1)
        setDx(Vec3((xmax-xmin)/(nx.getValue()-1),0,0));
    else
        setDx(Vec3(0,0,0));
    if (ny.getValue()>1)
        setDy(Vec3(0,(ymax-ymin)/(ny.getValue()-1),0));
    else
        setDy(Vec3(0,0,0));
    if (nz.getValue()>1)
        setDz(Vec3(0,0,(zmax-zmin)/(nz.getValue()-1)));
    else
        setDz(Vec3(0,0,0));
}

CubeTopology::Vec3 CubeTopology::getPoint(int i) const
{
    const int nx = this->nx.getValue();
    const int ny = this->ny.getValue();
    const int nz = this->nz.getValue();
    int x,y,z;
    if (!internalPoints.getValue() && (nx>1 && ny>1 && nz>1))
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

CubeTopology::Vec3 CubeTopology::getPoint(int x, int y, int z) const
{
    return p0+dx*x+dy*y+dz*z;
}

} // namespace topology

} // namespace component

} // namespace sofa
