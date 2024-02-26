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
#ifndef SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_H
#define SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_H
#include <SofaSphFluid/config.h>

#include <SofaSphFluid/SPHFluidForceField.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/VecTypes.h>

#include <vector>


namespace sofa::component::mapping
{



template <class InDataTypes, class OutDataTypes>
class SPHFluidSurfaceMappingGridTypes : public sofa::component::container::SpatialGridTypes<InDataTypes>
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
                val += (typename OutDataTypes::Real) (a*a*a);
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
    enum { GRIDDIM_LOG2 = 3 };
};


template <class In, class Out>
class SPHFluidSurfaceMapping : public core::Mapping<In, Out>, public topology::container::constant::MeshTopology
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(SPHFluidSurfaceMapping, In, Out), SOFA_TEMPLATE2(core::Mapping, In, Out), topology::container::constant::MeshTopology);

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
protected:
    SPHFluidSurfaceMapping();

    ~SPHFluidSurfaceMapping() override
    {}
public:
    double getStep() const;
    void setStep(double val);

    double getRadius() const;
    void setRadius(double val);

    double getIsoValue() const;
    void setIsoValue(double val);

    void init() override;

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn) override;

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& dOut, const Data<OutVecDeriv>& dIn) override;

    void draw(const core::visual::VisualParams* vparams) override;


protected:
    Data< double > d_mStep; ///< Step
    Data< double > d_mRadius; ///< Radius
    Data< double > d_mIsoValue; ///< Iso Value


    typedef forcefield::SPHFluidForceField<In> SPHForceField;
    SPHForceField* sph;

    // Marching cube data

    typedef SPHFluidSurfaceMappingGridTypes<In, Out> GridTypes;

    typedef sofa::component::container::SpatialGrid<GridTypes> Grid;
    typedef typename Grid::Cell Cell;
    typedef typename Grid::Grid SubGrid;
    typedef typename Grid::Key SubKey;
    typedef std::pair<SubKey,SubGrid*> GridEntry;
    enum { GRIDDIM = Grid::GRIDDIM };
    enum { DX = Grid::DX };
    enum { DY = Grid::DY };
    enum { DZ = Grid::DZ };

    Grid* grid;

    bool firstApply;

    void createPoints(OutVecCoord& out, OutVecDeriv* normals, const GridEntry& g, int x, int y, int z, Cell* c, const Cell* cx1, const Cell* cy1, const Cell* cz1, const OutReal isoval);

    void createFaces(OutVecCoord& out, OutVecDeriv* normals, const Cell** cells, const OutReal isoval);

    OutReal getValue(const SubGrid* g, int cx, int cy, int cz);

    OutDeriv calcGrad(const GridEntry& g, int x, int y, int z);

    template<int C>
    int addPoint(OutVecCoord& out, OutVecDeriv* normals, const GridEntry& g, int x,int y,int z, OutReal v0, OutReal v1, OutReal iso)
    {
        int p = int(out.size());
        OutCoord pos = OutCoord((OutReal)x,(OutReal)y,(OutReal)z);
        OutReal interp = (iso-v0)/(v1-v0);
        pos[C] += interp;
        out.resize(p+1);
        out[p] = pos * d_mStep.getValue();
        if (normals)
        {
            normals->resize(p+1);
            OutDeriv& n = (*normals)[p];
            OutDeriv n0 = calcGrad(g, x,y,z);
            OutDeriv n1 = calcGrad(g, (C==0)?x+1:x,(C==1)?y+1:y,(C==2)?z+1:z);
            n = n0 + (n1-n0) * interp;
            n.normalize();
            
            // TODO epernod 2019-10-14: remove this HACK: normals are going inside the model, certainly due to recent changes in cube orientation.
            n *= -1;
        }
        return p;
    }

    int addFace(int p1, int p2, int p3, int nbp);

public:
    bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }
};


#if  !defined(SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_CPP)  //// ATTENTION PB COMPIL WIN3Z
extern template class SOFA_SPH_FLUID_API SPHFluidSurfaceMapping< defaulttype::Vec3Types, defaulttype::Vec3Types >;



#endif


} // namespace sofa::component::mapping


#endif
