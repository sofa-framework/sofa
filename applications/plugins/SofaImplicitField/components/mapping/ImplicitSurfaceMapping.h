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
#pragma once
#include <SofaImplicitField/config.h>

#include <sofa/core/Mapping.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/helper/MarchingCubeUtility.h>
#include <SofaImplicitField/MarchingCube.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofaimplicitfield::mapping
{

using namespace sofa;
using sofa::component::topology::container::constant::MeshTopology;

template <class In, class Out>
class ImplicitSurfaceMapping : public core::Mapping<In, Out>, public MeshTopology
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(ImplicitSurfaceMapping, In, Out), SOFA_TEMPLATE2(core::Mapping, In, Out),  MeshTopology);

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

    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;
protected:
    ImplicitSurfaceMapping()
        : Inherit(),
          mStep(initData(&mStep,0.5,"step","Step")),
          mRadius(initData(&mRadius,2.0,"radius","Radius")),
          mIsoValue(initData(&mIsoValue,0.5,"isoValue","Iso Value")),
          mGridMin(initData(&mGridMin,InCoord(-100,-100,-100),"min","Grid Min")),
          mGridMax(initData(&mGridMax,InCoord(100,100,100),"max","Grid Max"))
    {
    }

    ~ImplicitSurfaceMapping() override
    {
    }
public:
    void init() override;

    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    double getStep() const { return mStep.getValue(); }
    void setStep(double val) { mStep.setValue(val); }

    double getRadius() const { return mRadius.getValue(); }
    void setRadius(double val) { mRadius.setValue(val); }

    double getIsoValue() const { return mIsoValue.getValue(); }
    void setIsoValue(double val) { mIsoValue.setValue(val); }

    const InCoord& getGridMin() const { return mGridMin.getValue(); }
    void setGridMin(const InCoord& val) { mGridMin.setValue(val); }
    void setGridMin(double x, double y, double z) { mGridMin.setValue( InCoord((InReal)x,(InReal)y,(InReal)z)); }

    const InCoord& getGridMax() const { return mGridMax.getValue(); }
    void setGridMax(const InCoord& val) { mGridMax.setValue(val); }
    void setGridMax(double x, double y, double z) { mGridMax.setValue( InCoord((InReal)x,(InReal)y,(InReal)z)); }

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;
    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;
    void applyJT( const sofa::core::MechanicalParams* /*mparams*/, InDataVecDeriv& /*out*/, const OutDataVecDeriv& /*in*/) override
    {
        msg_error() << "applyJT(dx) is not implemented";
    }

    void applyJT( const sofa::core::ConstraintParams* /*cparams*/, InDataMatrixDeriv& /*out*/, const OutDataMatrixDeriv& /*in*/) override
    {
        msg_error() << "applyJT(constraint) is not implemented";
    }

    void draw(const core::visual::VisualParams* params) override;

protected:
    Data <double > mStep; ///< Step
    Data <double > mRadius; ///< Radius
    Data <double > mIsoValue; ///< Iso Value

    Data< InCoord > mGridMin; ///< Grid Min
    Data< InCoord > mGridMax; ///< Grid Max

    Vec3d mLocalGridMin; ///< Grid Min
    Vec3d mLocalGridMax; ///< Grid Max


    // Marching cube data

    /// For each cube, store the vertex indices on each 3 first edges, and the data value
    struct CubeData
    {
        int p[3];
        OutReal data;
        inline friend std::istream& operator >> ( std::istream& in, CubeData& c)
        {
            in >> c.p[0] >> c.p[1] >> c.p[2] >> c.data;

            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const CubeData& c)
        {
            out << c.p[0] << " " << c.p[1] << " " << c.p[2] << " " << c.data ;
            return out;
        }
    };

    Data < sofa::type::vector<CubeData> > planes;
    typename sofa::type::vector<CubeData>::iterator P0; /// Pointer to first plane
    typename sofa::type::vector<CubeData>::iterator P1; /// Pointer to second plane
public:
    bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }

private:
    MarchingCube marchingCube;
};

#if  !defined(SOFA_COMPONENT_MAPPING_IMPLICITSURFACEMAPPING_CPP)
extern template class SOFA_SOFAIMPLICITFIELD_API ImplicitSurfaceMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
#endif

} // namespace

