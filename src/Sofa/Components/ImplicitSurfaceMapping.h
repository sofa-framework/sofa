#ifndef SOFA_COMPONENTS_IMPLICITSURFACEMAPPING_H
#define SOFA_COMPONENTS_IMPLICITSURFACEMAPPING_H

#include "Sofa/Core/Mapping.h"
#include "Sofa/Core/MechanicalModel.h"
#include "MeshTopology.h"
#include <vector>

namespace Sofa
{

namespace Components
{

using namespace Core;

template <class In, class Out>
class ImplicitSurfaceMapping : public Mapping<In, Out>, public MeshTopology
{
public:
    typedef Mapping<In, Out> Inherit;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename OutCoord::value_type OutReal;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename InCoord::value_type InReal;

    ImplicitSurfaceMapping(In* from, Out* to)
        : Inherit(from, to), mStep(0.5), mRadius(2.0), mIsoValue(0.5), mGridMin(-100,-100,-100), mGridMax(100, 100, 100)
    {
    }

    virtual ~ImplicitSurfaceMapping()
    {
    }

    double getStep() const { return mStep; }
    void setStep(double val) { mStep = val; }

    double getRadius() const { return mRadius; }
    void setRadius(double val) { mRadius = val; }

    double getIsoValue() const { return mIsoValue; }
    void setIsoValue(double val) { mIsoValue = val; }

    const InCoord& getGridMin() const { return mGridMin; }
    void setGridMin(const InCoord& val) { mGridMin = val; }
    void setGridMin(double x, double y, double z) { mGridMin = InCoord((InReal)x,(InReal)y,(InReal)z); }

    const InCoord& getGridMax() const { return mGridMax; }
    void setGridMax(const InCoord& val) { mGridMax = val; }
    void setGridMax(double x, double y, double z) { mGridMax = InCoord((InReal)x,(InReal)y,(InReal)z); }

    void apply( OutVecCoord& out, const InVecCoord& in );

    void applyJ( OutVecDeriv& out, const InVecDeriv& in );

    //void applyJT( InVecDeriv& out, const OutVecDeriv& in );
protected:
    double mStep;
    double mRadius;
    double mIsoValue;

    InCoord mGridMin;
    InCoord mGridMax;

    // Marching cube data

    /// For each cube, store the vertex indices on each 3 first edges, and the data value
    struct CubeData
    {
        int p[3];
        OutReal data;
    };

    std::vector<CubeData> planes;
    typename std::vector<CubeData>::iterator P0; /// Pointer to first plane
    typename std::vector<CubeData>::iterator P1; /// Pointer to second plane

    void newPlane();

    template<int C>
    int addPoint(OutVecCoord& out, int x,int y,int z, OutReal v0, OutReal v1, OutReal iso)
    {
        int p = out.size();
        OutCoord pos = OutCoord((OutReal)x,(OutReal)y,(OutReal)z);
        pos[C] -= (iso-v0)/(v1-v0);
        out.resize(p+1);
        out[p] = pos * mStep;
        return p;
    }

    int addFace(int p1, int p2, int p3, int nbp)
    {
        SeqTriangles& triangles = *seqTriangles.beginEdit();
        if ((unsigned)p1<(unsigned)nbp &&
            (unsigned)p2<(unsigned)nbp &&
            (unsigned)p3<(unsigned)nbp)
        {
            int f = triangles.size();
            triangles.push_back(Triangle(p1, p3, p2));
            return f;
        }
        else
        {
            std::cerr << "ERROR: Invalid face "<<p1<<" "<<p2<<" "<<p3<<std::endl;
            return -1;
        }
        seqTriangles.endEdit();
    }

};

// MARCHING CUBE TABLES
// ( table copied from an article of Paul Bourke )
// based on code by Cory Gene Bloyd

/* Convention:

         Z
         ^
         |
         4----4----5
        /|        /|
       7 |       5 |
      /  8      /  9
     7---+6----6   |
     |   |     |   |
     |   0----0+---1--> X
    11  /     10  /
     | 3       | 1
     |/        |/
     3----2----2
    /
   /
 |_
Y

*/

extern const int MarchingCubeEdgeTable[256];
extern const int MarchingCubeTriTable[256][16];

} // namespace Components

} // namespace Sofa

#endif
