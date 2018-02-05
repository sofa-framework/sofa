/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_BEHAVIORMODEL_EULERIANFLUID_GRID3D_H
#define SOFA_COMPONENT_BEHAVIORMODEL_EULERIANFLUID_GRID3D_H
#include "config.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/rmath.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace behaviormodel
{

namespace eulerianfluid
{

#ifndef NDEBUG
#define DEBUGGRID
#endif

class SOFA_EULERIAN_FLUID_API Grid3D
{
public:

    typedef float real;
    typedef sofa::defaulttype::Vec<3,real> vec3;

    struct Cell
    {
        vec3 u; ///< velocity (unit = cell width)
        int type; ///< First particle (or 0 for non-fluid cell)
        void clear()
        {
            u.clear();
            type = 0;
        }
    };

    int nx,ny,nz,nxny,ncell;

    enum { PART_EMPTY=0, PART_WALL=-1, PART_FULL=1 };

    Cell* fdata;
    real* pressure;
    real* levelset;
    //char* distance;

    real t;
    real tend;

    real max_pressure;
    Cell bcell;

    vec3 gravity;

    static const unsigned long* obstacles;

    Grid3D();
    ~Grid3D();

    void clear(int _nx,int _ny,int _nz);

    template<class T>
    static inline T lerp(T a, T b, real f) { return a+(b-a)*f; }

    int clamp_x(int x)
    {
        return (x<0)?0:(x>=nx)?nx-1:x;
    }

    int clamp_y(int y)
    {
        return (y<0)?0:(y>=ny)?ny-1:y;
    }

    int clamp_z(int z)
    {
        return (z<0)?0:(z>=nz)?nz-1:z;
    }

    int clamp_in_x(int x)
    {
        return (x<1)?1:(x>=nx-1)?nx-2:x;
    }

    int clamp_in_y(int y)
    {
        return (y<1)?1:(y>=ny-1)?ny-2:y;
    }

    int clamp_in_z(int z)
    {
        return (z<1)?1:(z>=nz-1)?nz-2:z;
    }

    int index(int x, int y, int z) const
    {
        return x + y*nx + z*nxny;
    }

    int index(const vec3& p) const
    {
        return index(sofa::helper::rnear(p[0]), sofa::helper::rnear(p[1]), sofa::helper::rnear(p[2]));
    }

    Cell* get(int x, int y, int z, Cell* base) const
    {
        return base+index(x,y,z);
    }

    const Cell* get(int x, int y, int z, const Cell* base) const
    {
        return base+index(x,y,z);
    }

    Cell* get(int x, int y, int z)
    {
#ifdef DEBUGGRID
        if (((unsigned)x>=(unsigned)nx) || ((unsigned)y>=(unsigned)ny) || ((unsigned)z>=(unsigned)nz))
        {
            msg_info("Grid3D")<<"INVALID CELL "<<x<<','<<y<<','<<z;
            return &bcell;
        }
#endif
        return get(x,y,z,fdata);
    }

    const Cell* get(int x, int y, int z) const
    {
#ifdef DEBUGGRID
        if (((unsigned)x>=(unsigned)nx) || ((unsigned)y>=(unsigned)ny) || ((unsigned)z>=(unsigned)nz))
        {
            msg_info("Grid3D")<<"INVALID CELL "<<x<<','<<y<<','<<z;
            return &bcell;
        }
#endif
        return get(x,y,z,fdata);
    }

    Cell* get(const vec3& p)
    {
        return get(sofa::helper::rnear(p[0]),sofa::helper::rnear(p[1]),sofa::helper::rnear(p[2]));
    }

    const Cell* get(const vec3& p) const
    {
        return get(sofa::helper::rnear(p[0]),sofa::helper::rnear(p[1]),sofa::helper::rnear(p[2]));
    }

    template<int C> real interp(const Cell* base, real fx, real fy, real fz) const
    {
        return lerp( lerp( lerp(get(0,0,0,base)->u[C],get(1,0,0,base)->u[C],fx),
                lerp(get(0,1,0,base)->u[C],get(1,1,0,base)->u[C],fx), fy ),
                lerp( lerp(get(0,0,1,base)->u[C],get(1,0,1,base)->u[C],fx),
                        lerp(get(0,1,1,base)->u[C],get(1,1,1,base)->u[C],fx), fy ),
                fz );
    }

    template<int C> real interp(vec3 p) const
    {
        p[C] += 0.5;
        int ix = sofa::helper::rfloor(p[0]);
        int iy = sofa::helper::rfloor(p[1]);
        int iz = sofa::helper::rfloor(p[2]);
        const Cell* base = get(ix,iy,iz);
        return interp<C>(base, p[0]-ix, p[1]-iy,p[2]-iz);
    }

    vec3 interp(vec3 p) const
    {
        return vec3( interp<0>(p), interp<1>(p), interp<2>(p) );
    }

    template<int C> void impulse(Cell* base, real fx, real fy, real fz, real i)
    {
        get(0,0,0,base)->u[C] += i*(1-fx)*(1-fy)*(1-fz);
        get(1,0,0,base)->u[C] += i*(  fx)*(1-fy)*(1-fz);
        get(0,1,0,base)->u[C] += i*(1-fx)*(  fy)*(1-fz);
        get(1,1,0,base)->u[C] += i*(  fx)*(  fy)*(1-fz);
        get(0,0,1,base)->u[C] += i*(1-fx)*(1-fy)*(  fz);
        get(1,0,1,base)->u[C] += i*(  fx)*(1-fy)*(  fz);
        get(0,1,1,base)->u[C] += i*(1-fx)*(  fy)*(  fz);
        get(1,1,1,base)->u[C] += i*(  fx)*(  fy)*(  fz);
    }

    template<int C> void impulse(vec3 p, real i)
    {
        p[C] += 0.5;
        int ix = sofa::helper::rfloor(p[0]);
        int iy = sofa::helper::rfloor(p[1]);
        int iz = sofa::helper::rfloor(p[2]);
        Cell* base = get(ix,iy,iz);
        impulse<C>(base, p[0]-ix, p[1]-iy,p[2]-iz, i);
    }

    void impulse(const vec3& p, const vec3& i)
    {
        impulse<0>(p,i[0]);
        impulse<1>(p,i[1]);
        impulse<2>(p,i[2]);
    }

    real* getpressure(int x, int y, int z)
    {
        return pressure + index(x,y,z);
    }

    real getpressure(vec3 p)
    {
        int ix = sofa::helper::rfloor(p[0]);
        int iy = sofa::helper::rfloor(p[1]);
        int iz = sofa::helper::rfloor(p[2]);
        real fx = p[0]-ix;
        real fy = p[1]-iy;
        real fz = p[2]-iz;
        real* base = getpressure(ix,iy,iz);
        return lerp( lerp( lerp(base[index(0,0,0)],base[index(1,0,0)],fx),
                lerp(base[index(0,1,0)],base[index(1,1,0)],fx), fy ),
                lerp( lerp(base[index(0,0,1)],base[index(1,0,1)],fx),
                        lerp(base[index(0,1,1)],base[index(1,1,1)],fx), fy ),
                fz );
    }

    real* getlevelset(int x, int y, int z)
    {
        return levelset + index(x,y,z);
    }

    real getlevelset(vec3 p)
    {
        int ix = sofa::helper::rfloor(p[0]);
        int iy = sofa::helper::rfloor(p[1]);
        int iz = sofa::helper::rfloor(p[2]);
        real fx = p[0]-ix;
        real fy = p[1]-iy;
        real fz = p[2]-iz;
        real* base = getlevelset(ix,iy,iz);
        return lerp( lerp( lerp(base[index(0,0,0)],base[index(1,0,0)],fx),
                lerp(base[index(0,1,0)],base[index(1,1,0)],fx), fy ),
                lerp( lerp(base[index(0,0,1)],base[index(1,0,1)],fx),
                        lerp(base[index(0,1,1)],base[index(1,1,1)],fx), fy ),
                fz );
    }

    void seed(real height);

    void seed(real height, vec3 normal);

    void seed(vec3 p0, vec3 p1, vec3 velocity=vec3(0,0,0));

    void step(Grid3D* prev, Grid3D* temp, real dt=0.04, real diff=0.00001);
    void step_init(const Grid3D* prev, Grid3D* temp, real dt, real diff);
    //void step_particles(Grid3D* prev, Grid3D* temp, real dt, real diff);
    void step_levelset(Grid3D* prev, Grid3D* temp, real dt, real diff);
    void step_forces(const Grid3D* prev, Grid3D* temp, real dt, real diff, real scale=1.0);
    void step_surface(const Grid3D* prev, Grid3D* temp, real dt, real diff);
    void step_advect(const Grid3D* prev, Grid3D* temp, real dt, real diff);
    void step_diffuse(const Grid3D* prev, Grid3D* temp, real dt, real diff);
    void step_project(const Grid3D* prev, Grid3D* temp, real dt, real diff);
    void step_color(const Grid3D* prev, Grid3D* temp, real dt, real diff);

    // internal helper function
    //  template<int C> inline real find_velocity(int x, int y, int z, int ind, int ind2, const Grid3D* prev, const Grid3D* temp);

    // Fast Marching Method Levelset Update
    enum Status { FMM_FRONT0 = 0, FMM_FAR = -1, FMM_KNOWN = -2, FMM_BORDER = -3 };
    int* fmm_status;
    int* fmm_heap;
    int fmm_heap_size;

    int fmm_pop();
    void fmm_push(int index);
    void fmm_swap(int entry1, int entry2);
};

} // namespace eulerianfluid

} // namespace behaviormodel

} // namespace component

} // namespace sofa

#endif
