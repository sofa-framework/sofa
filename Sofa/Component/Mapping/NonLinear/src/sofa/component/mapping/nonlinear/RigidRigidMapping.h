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
#include <sofa/component/mapping/nonlinear/config.h>

#include <sofa/core/Mapping.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#ifdef SOFA_BUILD_SOFA_COMPONENT_MAPPING_NONLINEAR
#define SOFA_ATTRIBUTE_DEPRECATED__RIGIDRIGIDMAPPING()
#else
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v23.06", "v23.12")
#define SOFA_ATTRIBUTE_DEPRECATED__RIGIDRIGIDMAPPING() \
    SOFA_ATTRIBUTE_DEPRECATED( \
    "v23.06", "v23.12", "Please use RigidMapping with template='Rigid3,Rigid3' instead. If this component is crucial to you please report that to sofa-dev@ so we can reconsider this component for future re-integration.")

#endif

// This component has been DEPRECATED since SOFA v23.06 and will be removed in SOFA v23.12.
// Please use RigidMapping with template='Rigid3,Rigid3' instead.
// If this component is crucial to you please report that to sofa-dev@ so we can reconsider this component for future re-integration.

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
class SOFA_ATTRIBUTE_DEPRECATED__RIGIDRIGIDMAPPING()
RigidRigidMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidRigidMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef Out OutDataTypes;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename Out::Coord::value_type Real;
    enum { N=OutDataTypes::spatial_dimensions };
    typedef type::Mat<N,N,Real> Mat;
    typedef type::Vec<N,Real> Vector ;

protected:
    Data < OutVecCoord > d_points; ///< Initial position of the points
    OutVecCoord m_pointsR0;
    Mat m_rotation;
    class Loader;
    void load(const char* filename);
    /// number of child frames per parent frame.
    /// If empty, all the children are attached to the parent with index
    /// given in the "index" attribute. If one value, each parent frame drives
    /// the given number of children frames. Otherwise, the values are the number
    /// of child frames driven by each parent frame.
    Data< sofa::type::vector<sofa::Size> >  d_repartition;

public:
    Data<sofa::Index> d_index; ///< input frame index
    sofa::core::objectmodel::DataFileName d_fileRigidRigidMapping; ///< Filename

    /// axis length for display
    Data<double> d_axisLength; ///< axis length for display
    Data< bool > d_indexFromEnd; ///< input DOF index starts from the end of input DOFs vector
    Data< bool > d_globalToLocalCoords; ///< are the output DOFs initially expressed in global coordinates

protected:
    RigidRigidMapping() ;
    ~RigidRigidMapping() override {}

public:
    void init() override;

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

    void computeAccFromMapping(const core::MechanicalParams *mparams, Data<OutVecDeriv>& acc_out, const Data<InVecDeriv>& v_in, const Data<InVecDeriv>& acc_in) override;

    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce ) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override
    {
        return nullptr;
    }

    void draw(const core::visual::VisualParams* vparams) override;

    void clear();

    void globalToLocalCoords(OutCoord& result, const InCoord &xfrom, const OutCoord &x);
    sofa::type::vector<sofa::Size> getRepartition() {return d_repartition.getValue(); }

    void setRepartition(sofa::Size value);
    void setRepartition(sofa::type::vector<sofa::Size> values);

protected:

    bool getShow(const core::objectmodel::BaseObject* /*m*/, const core::visual::VisualParams* vparams) const;
    bool getShow(const core::BaseMapping* /*m*/, const core::visual::VisualParams* vparams) const;
};

#if  !defined(SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidRigidMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types >;
#endif

} // namespace sofa::component::mapping::nonlinear
