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

#include <sofa/component/mapping/linear/config.h>

#include <sofa/core/Multi2Mapping.h>

namespace sofa::component::mapping::linear
{

/** \brief Maps a deformable mechanical state to another deformable mechanical state mapped onto a rigid frame.
 *  Inputs: One Vec3 and One Rigid  mechanical objects
 *  Output: One Vec3 mechanical object
 */

template<class InDataTypes, class OutDataTypes>
class DeformableOnRigidFrameMappingInternalData
{
public:
};

template <class TIn, class TInRoot, class TOut>
class DeformableOnRigidFrameMapping : public core::Multi2Mapping<TIn, TInRoot, TOut>
{
 public:
    SOFA_CLASS(SOFA_TEMPLATE3(DeformableOnRigidFrameMapping, TIn, TInRoot, TOut), SOFA_TEMPLATE3(core::Multi2Mapping, TIn, TInRoot, TOut) );

    typedef core::Multi2Mapping<TIn, TInRoot, TOut> Inherit;

    typedef TIn In;
    typedef TInRoot InRoot;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename OutCoord::value_type OutReal;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::Real InReal;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef typename InRoot::VecCoord InRootVecCoord;
    typedef typename InRoot::VecDeriv InRootVecDeriv;
    typedef typename InRoot::MatrixDeriv InRootMatrixDeriv;
    typedef typename InRoot::Coord InRootCoord;
    typedef typename InRoot::Deriv InRootDeriv;
    typedef typename InRoot::Real InRootReal;
    typedef Data<InRootVecCoord> InRootDataVecCoord;
    typedef Data<InRootVecDeriv> InRootDataVecDeriv;
    typedef Data<InRootMatrixDeriv> InRootDataMatrixDeriv;

    typedef typename OutCoord::value_type Real;
    typedef OutCoord Coord;
    typedef OutDeriv Deriv;
    enum { N=Out::spatial_dimensions };
    typedef type::Mat<N,N,Real> Mat;
    typedef type::Vec<N,Real> Vector ;

    OutVecCoord rotatedPoints;
    DeformableOnRigidFrameMappingInternalData<In, Out> data;
    Data<unsigned int> index; ///< input DOF index
    Data< bool > indexFromEnd; ///< input DOF index starts from the end of input DOFs vector
    Data<sofa::type::vector<unsigned int> >  repartition; ///< number of dest dofs per entry dof
    Data< bool > globalToLocalCoords; ///< are the output DOFs initially expressed in global coordinates

    Data< Real > m_rootAngularForceScaleFactor; ///< Scale factor applied on the angular force accumulated on the rigid model
    Data< Real > m_rootLinearForceScaleFactor; ///< Scale factor applied on the linear force accumulated on the rigid model

    int addPoint ( const OutCoord& c );
    int addPoint ( const OutCoord& c, int indexFrom );

    void init() override;

	void handleTopologyChange(core::topology::Topology* t) override;

    /// Return true if the destination model has the same topology as the source model.
    ///
    /// This is the case for mapping keeping a one-to-one correspondance between
    /// input and output DOFs (mostly identity or data-conversion mappings).
    bool sameTopology() const override { return true; }

    using Inherit::apply;
    using Inherit::applyJ;
    using Inherit::applyJT;

    //Apply
    void apply( OutVecCoord& out, const InVecCoord& in, const InRootVecCoord* inroot  );
    void apply(
        const core::MechanicalParams* /* mparams */, const type::vector<OutDataVecCoord*>& dataVecOutPos,
        const type::vector<const InDataVecCoord*>& dataVecInPos ,
        const type::vector<const InRootDataVecCoord*>& dataVecInRootPos) override;

    //ApplyJ
    void applyJ( OutVecDeriv& out, const InVecDeriv& in, const InRootVecDeriv* inroot );
    void applyJ(
        const core::MechanicalParams* /* mparams */, const type::vector< OutDataVecDeriv*>& dataVecOutVel,
        const type::vector<const InDataVecDeriv*>& dataVecInVel,
        const type::vector<const InRootDataVecDeriv*>& dataVecInRootVel) override;

    //ApplyJT Force
    void applyJT( InVecDeriv& out, const OutVecDeriv& in, InRootVecDeriv* outroot );
    void applyJT(
        const core::MechanicalParams* /* mparams */, const type::vector< InDataVecDeriv*>& dataVecOutForce,
        const type::vector< InRootDataVecDeriv*>& dataVecOutRootForce,
        const type::vector<const OutDataVecDeriv*>& dataVecInForce) override;

    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId inForce, core::ConstMultiVecDerivId outForce) override;


    //ApplyJT Constraint
    void applyJT( InMatrixDeriv& out, const OutMatrixDeriv& in, InRootMatrixDeriv* outroot );
    void applyJT(
        const core::ConstraintParams* /* cparams */, const type::vector< InDataMatrixDeriv*>& dataMatOutConst ,
        const type::vector< InRootDataMatrixDeriv*>&  dataMatOutRootConst ,
        const type::vector<const OutDataMatrixDeriv*>& dataMatInConst) override;

    /**
      * @brief
      MAP the mass: this function recompute the rigid mass (gravity center position and inertia) of the object
          based on its deformed shape
      */
    void recomputeRigidMass();

    void draw(const core::visual::VisualParams* vparams) override;

    void clear ( int reserve=0 );

    void setRepartition ( unsigned int value );
    void setRepartition ( sofa::type::vector<unsigned int> values );

protected:
    DeformableOnRigidFrameMapping();

    virtual ~DeformableOnRigidFrameMapping()
    {}

    core::State<In>* m_fromModel;
    core::State<Out>* m_toModel;
    core::State<InRoot>* m_fromRootModel;

    InRootCoord rootX;
};

#if !defined(SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAMEMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API DeformableOnRigidFrameMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types >;

#endif

} // namespace sofa::component::mapping::linear
