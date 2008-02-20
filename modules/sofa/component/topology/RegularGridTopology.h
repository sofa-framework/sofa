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
#ifndef SOFA_COMPONENT_TOPOLOGY_REGULARGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_REGULARGRIDTOPOLOGY_H

#include <sofa/component/topology/GridTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

class RegularGridTopology : public GridTopology
{
public:
    typedef Vec3d Vec3;

    RegularGridTopology(int nx, int ny, int nz);
    RegularGridTopology();

    virtual void reinit()
    {
        setPos(min.getValue()[0],max.getValue()[0],min.getValue()[1],max.getValue()[1],min.getValue()[2],max.getValue()[2]);

    }
    void parse(core::objectmodel::BaseObjectDescription* arg);

    void setP0(const Vec3& val) { p0 = val; }
    void setDx(const Vec3& val) { dx = val; inv_dx2 = 1/(dx*dx); }
    void setDy(const Vec3& val) { dy = val; inv_dy2 = 1/(dy*dy); }
    void setDz(const Vec3& val) { dz = val; inv_dz2 = 1/(dz*dz); }

    void setPos(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);


    const Vec3& getP0() const { return p0; }
    const Vec3& getDx() const { return dx; }
    const Vec3& getDy() const { return dy; }
    const Vec3& getDz() const { return dz; }

    unsigned getIndex( int i, int j, int k ) const; ///< one-dimensional index of a grid point
    Vec3 getPoint(int i) const;
    Vec3 getPoint(int x, int y, int z) const;
    bool hasPos()  const { return true; }
    double getPX(int i)  const { return getPoint(i)[0]; }
    double getPY(int i) const { return getPoint(i)[1]; }
    double getPZ(int i) const { return getPoint(i)[2]; }

    Vec3   getMin() const { return min.getValue();}
    Vec3   getMax() const { return max.getValue();}

    /// return the cube containing the given point (or -1 if not found).
    virtual int findCube(const Vec3& pos);
    int findHexa(const Vec3& pos) { return findCube(pos); }

    /// return the nearest cube (or -1 if not found).
    virtual int findNearestCube(const Vec3& pos);
    int findNearestHexa(const Vec3& pos) { return findNearestCube(pos); }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findCube(const Vec3& pos, double& fx, double &fy, double &fz);
    int findHexa(const Vec3& pos, double& fx, double &fy, double &fz) { return findCube(pos, fx, fy, fz); }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz);
    int findNearestHexa(const Vec3& pos, double& fx, double &fy, double &fz) { return findNearestCube(pos, fx, fy, fz); }

protected:
    Data< Vec3 > min, max;
    /// Position of point 0
    Vec3 p0;
    /// Distance between points in the grid. Must be perpendicular to each other
    Vec3 dx,dy,dz;
    double inv_dx2, inv_dy2, inv_dz2;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
