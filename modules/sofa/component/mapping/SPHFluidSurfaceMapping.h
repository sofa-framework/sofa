/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_H
#define SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/forcefield/SPHFluidForceField.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/helper/MarchingCubeUtility.h> // for marching cube tables
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::component::container;

template <class InDataTypes, class OutDataTypes>
class SPHFluidSurfaceMappingGridTypes : public SpatialGridTypes<InDataTypes>
{
public:
    typedef forcefield::SPHFluidForceField<InDataTypes> ParticleField;
    typedef typename InDataTypes::Real Real;
    /// For each cell, store the vertex indices on each 3 first edges, and the data value at the first corner
    class CellData
    {
    public:
        int p[3];
        typename OutDataTypes::Real val;
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
            if (field)
                val += (typename OutDataTypes::Real) field->getParticleField(i, r2/h2);
            else
            {
                Real a = 1-r2/h2;
                val += (a*a*a);
            }
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

template <class In, class Out>
class SPHFluidSurfaceMapping : public core::Mapping<In, Out>, public topology::MeshTopology
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
        : Inherit(from, to),
          mStep(initData(&mStep,0.5,"step","Step")),
          mRadius(initData(&mRadius,2.0,"radius","Radius")),
          mIsoValue(initData(&mIsoValue,0.5,"isoValue", "Iso Value")),
          sph(NULL), grid(NULL)
    {
    }

    virtual ~SPHFluidSurfaceMapping()
    {}

    double getStep() const
    {
        return mStep.getValue();
    }
    void setStep(double val)
    {
        mStep.setValue(val);
    }

    double getRadius() const
    {
        return mRadius.getValue();
    }
    void setRadius(double val)
    {
        mRadius.setValue(val);
    }

    double getIsoValue() const
    {
        return mIsoValue.getValue();
    }
    void setIsoValue(double val)
    {
        mIsoValue.setValue(val);
    }

    void init();

    void apply( OutVecCoord& out, const InVecCoord& in );

    void applyJ( OutVecDeriv& out, const InVecDeriv& in );

    //void applyJT( InVecDeriv& out, const OutVecDeriv& in );

    void draw();


protected:
    Data< double > mStep;
    Data< double > mRadius;
    Data< double > mIsoValue;

    typedef forcefield::SPHFluidForceField<typename In::DataTypes> SPHForceField;
    SPHForceField* sph;

    // Marching cube data

    typedef SPHFluidSurfaceMappingGridTypes<typename In::DataTypes, typename Out::DataTypes> GridTypes;

    typedef SpatialGrid<GridTypes> Grid;
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
        out[p] = pos * mStep.getValue();
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
            serr << "ERROR: Invalid face "<<p1<<" "<<p2<<" "<<p3<<sendl;
            return -1;
        }
    }

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
