/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONTAINER_DISTANCEGRIDCOLLISIONMODEL_H
#define SOFA_COMPONENT_CONTAINER_DISTANCEGRIDCOLLISIONMODEL_H

#include <sofa/SofaAdvanced.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/io/Mesh.h>

namespace sofa
{

namespace component
{

namespace container
{

class SOFA_VOLUMETRIC_DATA_API DistanceGrid
{
public:
    static SReal maxDist() { return (SReal)1e10; }
    typedef sofa::defaulttype::Vector3 Coord;
    typedef sofa::defaulttype::ExtVector<SReal> VecSReal;
    typedef sofa::defaulttype::ExtVector<Coord> VecCoord;

    DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax);

    DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax, defaulttype::ExtVectorAllocator<SReal>* alloc);

protected:
    ~DistanceGrid();

public:
    /// Load a distance grid
    static DistanceGrid* load(const std::string& filename, double scale=1.0, double sampling=0.0, int nx=64, int ny=64, int nz=64, Coord pmin = Coord(), Coord pmax = Coord());
    static DistanceGrid* loadVTKFile(const std::string& filename, double scale=1.0, double sampling=0.0);

    /// Load or reuse a distance grid
    static DistanceGrid* loadShared(const std::string& filename, double scale=1.0, double sampling=0.0, int nx=64, int ny=64, int nz=64, Coord pmin = Coord(), Coord pmax = Coord());

    /// Add one reference to this grid. Note that loadShared already does this.
    DistanceGrid* addRef();

    /// Release one reference, deleting this grid if this is the last
    bool release();

    /// Save current grid
    bool save(const std::string& filename);

    /// Compute distance field from given mesh
    void calcDistance(sofa::helper::io::Mesh* mesh, double scale=1.0);

    /// Compute distance field for a cube of the given half-size.
    /// Also create a mesh of points using np points per axis
    void calcCubeDistance(SReal dim=1, int np=5);

    /// Sample the surface with points approximately separated by the given sampling distance (expressed in voxels if the value is negative)
    void sampleSurface(double sampling=-1.0);

    /// Update bbox
    void computeBBox();

    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    const Coord& getCellWidth() const { return cellWidth; }

    int size() const { return nxnynz; }

    const Coord& getBBMin() const { return bbmin; }
    const Coord& getBBMax() const { return bbmax; }
    void setBBMin(const Coord& val) { bbmin = val; }
    void setBBMax(const Coord& val) { bbmax = val; }
    Coord getBBCorner(int i) const { return Coord((i&1)?bbmax[0]:bbmin[0],(i&2)?bbmax[1]:bbmin[1],(i&4)?bbmax[2]:bbmin[2]); }
    bool inBBox(const Coord& p, SReal margin=0.0f) const
    {
        for (int c=0; c<3; ++c)
            if (p[c] < bbmin[c]-margin || p[c] > bbmax[c]+margin) return false;
        return true;
    }

    const Coord& getPMin() const { return pmin; }
    const Coord& getPMax() const { return pmax; }
    Coord getCorner(int i) const { return Coord((i&1)?pmax[0]:pmin[0],(i&2)?pmax[1]:pmin[1],(i&4)?pmax[2]:pmin[2]); }

    bool isCube() const { return cubeDim != 0; }
    SReal getCubeDim() const { return cubeDim; }

    bool inGrid(const Coord& p) const
    {
        Coord epsilon = cellWidth*0.1;
        for (int c=0; c<3; ++c)
            if (p[c] < pmin[c]+epsilon[c] || p[c] > pmax[c]-epsilon[c]) return false;
        return true;
    }

    Coord clamp(Coord p) const
    {
        for (int c=0; c<3; ++c)
            if (p[c] < pmin[c]) p[c] = pmin[c];
            else if (p[c] > pmax[c]) p[c] = pmax[c];
        return p;
    }

    int ix(const Coord& p) const
    {
        return helper::rfloor((p[0]-pmin[0])*invCellWidth[0]);
    }

    int iy(const Coord& p) const
    {
        return helper::rfloor((p[1]-pmin[1])*invCellWidth[1]);
    }

    int iz(const Coord& p) const
    {
        return helper::rfloor((p[2]-pmin[2])*invCellWidth[2]);
    }

    int index(const Coord& p, Coord& coefs) const
    {
        coefs[0] = (p[0]-pmin[0])*invCellWidth[0];
        coefs[1] = (p[1]-pmin[1])*invCellWidth[1];
        coefs[2] = (p[2]-pmin[2])*invCellWidth[2];
        int x = helper::rfloor(coefs[0]);
        if (x<0) x=0; else if (x>=nx-1) x=nx-2;
        coefs[0] -= x;
        int y = helper::rfloor(coefs[1]);
        if (y<0) y=0; else if (y>=ny-1) y=ny-2;
        coefs[1] -= y;
        int z = helper::rfloor(coefs[2]);
        if (z<0) z=0; else if (z>=nz-1) z=nz-2;
        coefs[2] -= z;
        return x+nx*(y+ny*(z));
    }

    int index(const Coord& p) const
    {
        Coord coefs;
        return index(p, coefs);
    }

    int index(int x, int y, int z)
    {
        return x+nx*(y+ny*(z));
    }

    Coord coord(int x, int y, int z)
    {
        return pmin+Coord(x*cellWidth[0], y*cellWidth[1], z*cellWidth[2]);
    }

    SReal operator[](int index) const { return dists[index]; }
    SReal& operator[](int index) { return dists[index]; }

    static SReal interp(SReal coef, SReal a, SReal b)
    {
        return a+coef*(b-a);
    }

    SReal interp(int index, const Coord& coefs) const
    {
        return interp(coefs[2],interp(coefs[1],interp(coefs[0],dists[index          ],dists[index+1        ]),
                interp(coefs[0],dists[index  +nx     ],dists[index+1+nx     ])),
                interp(coefs[1],interp(coefs[0],dists[index     +nxny],dists[index+1   +nxny]),
                        interp(coefs[0],dists[index  +nx+nxny],dists[index+1+nx+nxny])));
    }

    SReal interp(const Coord& p) const
    {
        Coord coefs;
        int i = index(p, coefs);
        return interp(i, coefs);
    }

    Coord grad(int index, const Coord& coefs) const
    {
        // val = dist[0][0][0] * (1-x) * (1-y) * (1-z)
        //     + dist[1][0][0] * (  x) * (1-y) * (1-z)
        //     + dist[0][1][0] * (1-x) * (  y) * (1-z)
        //     + dist[1][1][0] * (  x) * (  y) * (1-z)
        //     + dist[0][0][1] * (1-x) * (1-y) * (  z)
        //     + dist[1][0][1] * (  x) * (1-y) * (  z)
        //     + dist[0][1][1] * (1-x) * (  y) * (  z)
        //     + dist[1][1][1] * (  x) * (  y) * (  z)
        // dval / dx = (dist[1][0][0]-dist[0][0][0]) * (1-y) * (1-z)
        //           + (dist[1][1][0]-dist[0][1][0]) * (  y) * (1-z)
        //           + (dist[1][0][1]-dist[0][0][1]) * (1-y) * (  z)
        //           + (dist[1][1][1]-dist[0][1][1]) * (  y) * (  z)
        const SReal dist000 = dists[index          ];
        const SReal dist100 = dists[index+1        ];
        const SReal dist010 = dists[index  +nx     ];
        const SReal dist110 = dists[index+1+nx     ];
        const SReal dist001 = dists[index     +nxny];
        const SReal dist101 = dists[index+1   +nxny];
        const SReal dist011 = dists[index  +nx+nxny];
        const SReal dist111 = dists[index+1+nx+nxny];
        return Coord(
                interp(coefs[2],interp(coefs[1],dist100-dist000,dist110-dist010),interp(coefs[1],dist101-dist001,dist111-dist011)), //*invCellWidth[0],
                interp(coefs[2],interp(coefs[0],dist010-dist000,dist110-dist100),interp(coefs[0],dist011-dist001,dist111-dist101)), //*invCellWidth[1],
                interp(coefs[1],interp(coefs[0],dist001-dist000,dist101-dist100),interp(coefs[0],dist011-dist010,dist111-dist110))); //*invCellWidth[2]);
    }

    Coord grad(const Coord& p) const
    {
        Coord coefs;
        int i = index(p, coefs);
        return grad(i, coefs);
    }

    SReal eval(const Coord& x) const
    {
        SReal d;
        if (inGrid(x))
        {
            d = interp(x);
        }
        else
        {
            Coord xclamp = clamp(x);
            d = interp(xclamp);
            d = helper::rsqrt((x-xclamp).norm2() + d*d); // we underestimate the distance
        }
        return d;
    }

    template<class T>
    T tgrad(const T& p) const
    {
        Coord cp;
        for (unsigned int i=0; i < cp.size() && i < p.size(); ++i) cp[i] = (SReal) p[i];
        Coord cr = grad(cp);
        T r;
        for (unsigned int i=0; i < cr.size() && i < p.size(); ++i) r[i] = (typename T::value_type) cr[i];
        return r;
    }

    template<class T>
    typename T::value_type teval(const T& x) const
    {
        Coord cx;
        for (unsigned int i=0; i < cx.size() && i < x.size(); ++i) cx[i] = (SReal) x[i];
        return (typename T::value_type) eval(cx);
    }

    SReal quickeval(const Coord& x) const
    {
        SReal d;
        if (inGrid(x))
        {
            d = dists[index(x)] - cellWidth[0]; // we underestimate the distance
        }
        else
        {
            Coord xclamp = clamp(x);
            d = dists[index(xclamp)] - cellWidth[0]; // we underestimate the distance
            d = helper::rsqrt((x-xclamp).norm2() + d*d);
        }
        return d;
    }

    SReal eval2(const Coord& x) const
    {
        SReal d2;
        if (inGrid(x))
        {
            SReal d = interp(x);
            d2 = d*d;
        }
        else
        {
            Coord xclamp = clamp(x);
            SReal d = interp(xclamp);
            d2 = ((x-xclamp).norm2() + d*d); // we underestimate the distance
        }
        return d2;
    }

    SReal quickeval2(const Coord& x) const
    {
        SReal d2;
        if (inGrid(x))
        {
            SReal d = dists[index(x)] - cellWidth[0]; // we underestimate the distance
            d2 = d*d;
        }
        else
        {
            Coord xclamp = clamp(x);
            SReal d = dists[index(xclamp)] - cellWidth[0]; // we underestimate the distance
            d2 = ((x-xclamp).norm2() + d*d);
        }
        return d2;
    }

    VecCoord meshPts;

protected:
    int nbRef;
    VecSReal dists;
    const int nx,ny,nz, nxny, nxnynz;
    const Coord pmin, pmax;
    const Coord cellWidth, invCellWidth;
    Coord bbmin, bbmax; ///< bounding box of the object, smaller than the grid

    SReal cubeDim; ///< Cube dimension (!=0 if this is actually a cube

    // Fast Marching Method Update
    enum Status { FMM_FRONT0 = 0, FMM_FAR = -1, FMM_KNOWN_OUT = -2, FMM_KNOWN_IN = -3 };
    helper::vector<int> fmm_status;
    helper::vector<int> fmm_heap;
    int fmm_heap_size;

    int fmm_pop();
    void fmm_push(int index);
    void fmm_swap(int entry1, int entry2);

    // Grid shared resources

    struct DistanceGridParams
    {
        std::string filename;
        double scale;
        double sampling;
        int nx,ny,nz;
        Coord pmin,pmax;
        bool operator==(const DistanceGridParams& v) const
        {
            if (!(filename == v.filename)) return false;
            if (!(scale    == v.scale   )) return false;
            if (!(sampling == v.sampling)) return false;
            if (!(nx       == v.nx      )) return false;
            if (!(ny       == v.ny      )) return false;
            if (!(nz       == v.nz      )) return false;
            if (!(pmin[0]  == v.pmin[0] )) return false;
            if (!(pmin[1]  == v.pmin[1] )) return false;
            if (!(pmin[2]  == v.pmin[2] )) return false;
            if (!(pmax[0]  == v.pmax[0] )) return false;
            if (!(pmax[1]  == v.pmax[1] )) return false;
            if (!(pmax[2]  == v.pmax[2] )) return false;
            return true;
        }
        bool operator<(const DistanceGridParams& v) const
        {
            if (filename < v.filename) return false;
            if (filename > v.filename) return true;
            if (scale    < v.scale   ) return false;
            if (scale    > v.scale   ) return true;
            if (sampling < v.sampling) return false;
            if (sampling > v.sampling) return true;
            if (nx       < v.nx      ) return false;
            if (nx       > v.nx      ) return true;
            if (ny       < v.ny      ) return false;
            if (ny       > v.ny      ) return true;
            if (nz       < v.nz      ) return false;
            if (nz       > v.nz      ) return true;
            if (pmin[0]  < v.pmin[0] ) return false;
            if (pmin[0]  > v.pmin[0] ) return true;
            if (pmin[1]  < v.pmin[1] ) return false;
            if (pmin[1]  > v.pmin[1] ) return true;
            if (pmin[2]  < v.pmin[2] ) return false;
            if (pmin[2]  > v.pmin[2] ) return true;
            if (pmax[0]  < v.pmax[0] ) return false;
            if (pmax[0]  > v.pmax[0] ) return true;
            if (pmax[1]  < v.pmax[1] ) return false;
            if (pmax[1]  > v.pmax[1] ) return true;
            if (pmax[2]  < v.pmax[2] ) return false;
            if (pmax[2]  > v.pmax[2] ) return true;
            return false;
        }
        bool operator>(const DistanceGridParams& v) const
        {
            if (filename > v.filename) return false;
            if (filename < v.filename) return true;
            if (scale    > v.scale   ) return false;
            if (scale    < v.scale   ) return true;
            if (sampling < v.sampling) return false;
            if (sampling > v.sampling) return true;
            if (nx       > v.nx      ) return false;
            if (nx       < v.nx      ) return true;
            if (ny       > v.ny      ) return false;
            if (ny       < v.ny      ) return true;
            if (nz       > v.nz      ) return false;
            if (nz       < v.nz      ) return true;
            if (pmin[0]  > v.pmin[0] ) return false;
            if (pmin[0]  < v.pmin[0] ) return true;
            if (pmin[1]  > v.pmin[1] ) return false;
            if (pmin[1]  < v.pmin[1] ) return true;
            if (pmin[2]  > v.pmin[2] ) return false;
            if (pmin[2]  < v.pmin[2] ) return true;
            if (pmax[0]  > v.pmax[0] ) return false;
            if (pmax[0]  < v.pmax[0] ) return true;
            if (pmax[1]  > v.pmax[1] ) return false;
            if (pmax[1]  < v.pmax[1] ) return true;
            if (pmax[2]  > v.pmax[2] ) return false;
            if (pmax[2]  < v.pmax[2] ) return true;
            return false;
        }
    };
    static std::map<DistanceGridParams, DistanceGrid*>& getShared();

};

} // namespace container

} // namespace component

} // namespace sofa

#endif
