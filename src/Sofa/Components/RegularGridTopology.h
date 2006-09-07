#ifndef SOFA_COMPONENTS_REGULARGRIDTOPOLOGY_H
#define SOFA_COMPONENTS_REGULARGRIDTOPOLOGY_H

#include "GridTopology.h"
#include "Common/Vec.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class RegularGridTopology : public GridTopology
{
public:
    typedef Vec3d Vec3;

    RegularGridTopology(int nx, int ny, int nz);

    void setP0(const Vec3& val) { p0 = val; }
    void setDx(const Vec3& val) { dx = val; inv_dx2 = 1/(dx*dx); }
    void setDy(const Vec3& val) { dy = val; inv_dy2 = 1/(dy*dy); }
    void setDz(const Vec3& val) { dz = val; inv_dz2 = 1/(dz*dz); }

    void setPos(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);

    const Vec3& getP0() const { return p0; }
    const Vec3& getDx() const { return dx; }
    const Vec3& getDy() const { return dy; }
    const Vec3& getDz() const { return dz; }

    Vec3 getPoint(int i) const ;
    Vec3 getPoint(int x, int y, int z) const ;
    bool hasPos() const  { return true; }
    double getPX(int i) const { return getPoint(i)[0]; }
    double getPY(int i) const { return getPoint(i)[1]; }
    double getPZ(int i) const { return getPoint(i)[2]; }

    /// return the cube containing the given point (or -1 if not found).
    virtual int findCube(const Vec3& pos) const;

    /// return the nearest cube (or -1 if not found).
    virtual int findNearestCube(const Vec3& pos);

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findCube(const Vec3& pos, double& fx, double &fy, double &fz) const;

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz);

protected:
    /// Position of point 0
    Vec3 p0;
    /// Distance between points in the grid. Must be perpendicular to each other
    Vec3 dx,dy,dz;
    double inv_dx2, inv_dy2, inv_dz2;
};

} // namespace Components

} // namespace Sofa

#endif
