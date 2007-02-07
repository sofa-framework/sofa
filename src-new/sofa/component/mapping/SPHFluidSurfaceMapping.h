#ifndef SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_H
#define SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/forcefield/SPHFluidForceField.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/mapping/ImplicitSurfaceMapping.h> // for marching cube tables
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

// TODO: move SpatialGridContainer to another namespace?
using namespace sofa::component::behaviormodel::eulerianfluid;

template <class In, class Out>
class SPHFluidSurfaceMapping : public core::Mapping<In, Out>, public topology::MeshTopology, public core::VisualModel
{
public:
    typedef core::Mapping<In, Out> Inherit;
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

    SPHFluidSurfaceMapping(In* from, Out* to)
        : Inherit(from, to), mStep(0.5), mRadius(2.0), mIsoValue(0.5), sph(NULL), grid(NULL)
    {}

    virtual ~SPHFluidSurfaceMapping()
    {}

    double getStep() const
    {
        return mStep;
    }
    void setStep(double val)
    {
        mStep = val;
    }

    double getRadius() const
    {
        return mRadius;
    }
    void setRadius(double val)
    {
        mRadius = val;
    }

    double getIsoValue() const
    {
        return mIsoValue;
    }
    void setIsoValue(double val)
    {
        mIsoValue = val;
    }

    void init();

    void apply( OutVecCoord& out, const InVecCoord& in );

    void applyJ( OutVecDeriv& out, const InVecDeriv& in );

    //void applyJT( InVecDeriv& out, const OutVecDeriv& in );

    // -- VisualModel interface
    void draw();
    void initTextures()
    { }
    void update()
    { }

protected:
    double mStep;
    double mRadius;
    double mIsoValue;

    typedef forcefield::SPHFluidForceField<typename In::DataTypes> SPHForceField;
    SPHForceField* sph;

    // Marching cube data

    class GridTypes : public SpatialGridContainerTypes<InCoord>
    {
    public:
        typedef SPHForceField ParticleField;
        typedef InReal Real;
        /// For each cell, store the vertex indices on each 3 first edges, and the data value at the first corner
        class CellData
        {
        public:
            int p[3];
            OutReal val;
            CellData()
            {
                clear();
            }
            void clear()
            {
                p[0]=p[1]=p[2]=-1;
                val=0;
            }
            void add
            (ParticleField* field, int i, Real r2, Real h2)
            {
                val += (OutReal) field->getParticleField(i, r2/h2);
            }
        };

        class GridData
        {
        public:
            bool visited;
            void clear()
            {
                visited = false;
            }
            GridData()
            {
                clear();
            }
        };
        enum { GRIDDIM_LOG2 = 2 };
    };

    typedef SpatialGridContainer<GridTypes> Grid;
    typedef typename Grid::Cell Cell;
    enum { GRIDDIM = Grid::GRIDDIM };
    enum { DX = Grid::DX };
    enum { DY = Grid::DY };
    enum { DZ = Grid::DZ };

    Grid* grid;

    void createPoints(OutVecCoord& out, int x, int y, int z, Cell* c, const Cell* cx, const Cell* cy, const Cell* cz, const OutReal isoval);

    void createFaces(OutVecCoord& out, const Cell** cells, const OutReal isoval);

    template<int C>
    int addPoint(OutVecCoord& out, int x,int y,int z, OutReal v0, OutReal v1, OutReal iso)
    {
        int p = out.size();
        OutCoord pos = OutCoord((OutReal)x,(OutReal)y,(OutReal)z);
        pos[C] += (iso-v0)/(v1-v0);
        out.resize(p+1);
        out[p] = pos * mStep;
        return p;
    }

    int addFace(int p1, int p2, int p3, int nbp)
    {
        if ((unsigned)p1<(unsigned)nbp &&
            (unsigned)p2<(unsigned)nbp &&
            (unsigned)p3<(unsigned)nbp)
        {
            SeqTriangles& triangles = *seqTriangles.beginEdit();
            int f = triangles.size();
            triangles.push_back(Triangle(p1, p3, p2));
            seqTriangles.endEdit();
            return f;
        }
        else
        {
            std::cerr << "ERROR: Invalid face "<<p1<<" "<<p2<<" "<<p3<<std::endl;
            return -1;
        }
    }

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
