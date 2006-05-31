#include "RegularGridTopology.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

void create(RegularGridTopology*& obj, ObjectDescription* arg)
{
    const char* nx = arg->getAttribute("nx");
    const char* ny = arg->getAttribute("ny");
    const char* nz = arg->getAttribute("nz");
    if (!nx || !ny || !nz)
    {
        std::cerr << "RegularGridTopology requires nx, ny and nz attributes\n";
    }
    else
    {
        obj = new RegularGridTopology(atoi(nx),atoi(ny),atoi(nz));
        const char* xmin = arg->getAttribute("xmin","0");
        const char* ymin = arg->getAttribute("ymin","0");
        const char* zmin = arg->getAttribute("zmin","0");
        const char* xmax = arg->getAttribute("xmax",nx);
        const char* ymax = arg->getAttribute("ymax",ny);
        const char* zmax = arg->getAttribute("zmax",nz);
        obj->setPos(atof(xmin),atof(xmax),atof(ymin),atof(ymax),atof(zmin),atof(zmax));
    }
}

SOFA_DECL_CLASS(RegularGridTopology)

Creator<ObjectFactory, RegularGridTopology> RegularGridTopologyClass("RegularGrid");

RegularGridTopology::RegularGridTopology()
{
}

RegularGridTopology::RegularGridTopology(int nx, int ny, int nz)
    : GridTopology(nx, ny, nz)
{
}

void RegularGridTopology::setPos(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
{
    setP0(Vec3(xmin,ymin,zmin));
    if (nx>1)
        setDx(Vec3((xmax-xmin)/(nx-1),0,0));
    else
        setDx(Vec3(0,0,0));
    if (ny>1)
        setDy(Vec3(0,(ymax-ymin)/(ny-1),0));
    else
        setDy(Vec3(0,0,0));
    if (nz>1)
        setDz(Vec3(0,0,(zmax-zmin)/(nz-1)));
    else
        setDz(Vec3(0,0,0));
}

RegularGridTopology::Vec3 RegularGridTopology::getPoint(int i)
{
    int x = i%nx; i/=nx;
    int y = i%ny; i/=ny;
    int z = i;
    return getPoint(x,y,z);
}

RegularGridTopology::Vec3 RegularGridTopology::getPoint(int x, int y, int z)
{
    return p0+dx*x+dy*y+dz*z;
}

/// return the cube containing the given point (or -1 if not found).
int RegularGridTopology::findCube(const Vec3& pos) const
{
    if (nx<2 || ny<2 || nz<2) return -1;
    Vec3 p = pos-p0;
    double x = p*dx*inv_dx2;
    double y = p*dy*inv_dy2;
    double z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if ((unsigned)ix<(unsigned)nx-2 && (unsigned)iy<(unsigned)ny-2 && (unsigned)iz<(unsigned)nz-2)
    {
        return cube(ix,iy,iz);
    }
    else
    {
        return -1;
    }
}

/// return the nearest cube (or -1 if not found).
int RegularGridTopology::findNearestCube(const Vec3& pos) const
{
    if (nx<2 || ny<2 || nz<2) return -1;
    Vec3 p = pos-p0;
    double x = p*dx*inv_dx2;
    double y = p*dy*inv_dy2;
    double z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (ix<0) ix=0; else if (ix>nx-2) ix=nx-2;
    if (iy<0) iy=0; else if (iy>ny-2) iy=ny-2;
    if (iz<0) iz=0; else if (iz>nz-2) iz=nz-2;
    return cube(ix,iy,iz);
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int RegularGridTopology::findCube(const Vec3& pos, double& fx, double &fy, double &fz) const
{
    if (nx<2 || ny<2 || nz<2) return -1;
    Vec3 p = pos-p0;
    double x = p*dx*inv_dx2;
    double y = p*dy*inv_dy2;
    double z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if ((unsigned)ix<(unsigned)nx-2 && (unsigned)iy<(unsigned)ny-2 && (unsigned)iz<(unsigned)nz-2)
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
int RegularGridTopology::findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz) const
{
    if (nx<2 || ny<2 || nz<2) return -1;
    Vec3 p = pos-p0;
    double x = p*dx*inv_dx2;
    double y = p*dy*inv_dy2;
    double z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    if (ix<0) ix=0; else if (ix>nx-2) ix=nx-2;
    if (iy<0) iy=0; else if (iy>ny-2) iy=ny-2;
    if (iz<0) iz=0; else if (iz>nz-2) iz=nz-2;
    fx = x-ix;
    fy = y-iy;
    fz = z-iz;
    return cube(ix,iy,iz);
}

} // namespace Components

} // namespace Sofa
