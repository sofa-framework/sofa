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
#include <ArticulatedSystemPlugin/config.h>

#include <sofa/core/Multi2Mapping.h>

#include <ArticulatedSystemPlugin/ArticulatedHierarchyContainer.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <vector>

namespace sofa::component::mapping
{

template <class TIn, class TInRoot, class TOut>
class ArticulatedSystemMapping : public core::Multi2Mapping<TIn, TInRoot, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(ArticulatedSystemMapping, TIn, TInRoot, TOut), SOFA_TEMPLATE3(core::Multi2Mapping, TIn, TInRoot, TOut));

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
protected:
    ArticulatedSystemMapping();

    ~ArticulatedSystemMapping() override
    {
    }
public:
    void init() override;
    void bwdInit() override;
    void reset() override;

    using Inherit::apply;
    using Inherit::applyJ;
    using Inherit::applyJT;

    //Apply
    void apply( OutVecCoord& out, const InVecCoord& in, const InRootVecCoord* inroot  );
    void apply(
        const core::MechanicalParams* /* mparams */, const type::vector<OutDataVecCoord*>& dataVecOutPos,
        const type::vector<const InDataVecCoord*>& dataVecInPos ,
        const type::vector<const InRootDataVecCoord*>& dataVecInRootPos) override
    {
        if (d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
            return;

        if(dataVecOutPos.empty() || dataVecInPos.empty())
            return;

        const InRootVecCoord* inroot = nullptr;

        //We need only one input In model and input Root model (if present)
        OutVecCoord& out = *dataVecOutPos[0]->beginEdit();
        const InVecCoord& in = dataVecInPos[0]->getValue();

        if (!dataVecInRootPos.empty())
            inroot = &dataVecInRootPos[0]->getValue();

        apply(out, in, inroot);

        dataVecOutPos[0]->endEdit();
    }

    //ApplyJ
    void applyJ( OutVecDeriv& out, const InVecDeriv& in, const InRootVecDeriv* inroot );
    void applyJ(
        const core::MechanicalParams* /* mparams */, const type::vector< OutDataVecDeriv*>& dataVecOutVel,
        const type::vector<const InDataVecDeriv*>& dataVecInVel,
        const type::vector<const InRootDataVecDeriv*>& dataVecInRootVel) override
    {
        if (d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
            return;

        if(dataVecOutVel.empty() || dataVecInVel.empty())
            return;

        const InRootVecDeriv* inroot = nullptr;

        //We need only one input In model and input Root model (if present)
        OutVecDeriv& out = *dataVecOutVel[0]->beginEdit();
        const InVecDeriv& in = dataVecInVel[0]->getValue();

        if (!dataVecInRootVel.empty())
            inroot = &dataVecInRootVel[0]->getValue();

        applyJ(out,in, inroot);

        dataVecOutVel[0]->endEdit();
    }

    //ApplyJT Force
    void applyJT( InVecDeriv& out, const OutVecDeriv& in, InRootVecDeriv* outroot );
    void applyJT(
        const core::MechanicalParams* /* mparams */, const type::vector< InDataVecDeriv*>& dataVecOutForce,
        const type::vector< InRootDataVecDeriv*>& dataVecOutRootForce,
        const type::vector<const OutDataVecDeriv*>& dataVecInForce) override
    {
        if (d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
            return;

        if(dataVecOutForce.empty() || dataVecInForce.empty())
            return;

        InRootVecDeriv* outroot = nullptr;

        //We need only one input In model and input Root model (if present)
        InVecDeriv& out = *dataVecOutForce[0]->beginEdit();
        const OutVecDeriv& in = dataVecInForce[0]->getValue();

        if (!dataVecOutRootForce.empty())
            outroot = dataVecOutRootForce[0]->beginEdit();

        applyJT(out,in, outroot);

        dataVecOutForce[0]->endEdit();
        if (outroot != nullptr)
            dataVecOutRootForce[0]->endEdit();
    }

    void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override
    {

    }



    //ApplyJT Constraint
    void applyJT( InMatrixDeriv& out, const OutMatrixDeriv& in, InRootMatrixDeriv* outroot );
    void applyJT(
        const core::ConstraintParams* /* cparams */, const type::vector< InDataMatrixDeriv*>& dataMatOutConst ,
        const type::vector< InRootDataMatrixDeriv*>&  dataMatOutRootConst ,
        const type::vector<const OutDataMatrixDeriv*>& dataMatInConst) override
    {
        if (d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
            return;

        if(dataMatOutConst.empty() || dataMatInConst.empty())
            return;

        InRootMatrixDeriv* outroot = nullptr;

        //We need only one input In model and input Root model (if present)
        InMatrixDeriv& out = *dataMatOutConst[0]->beginEdit();
        const OutMatrixDeriv& in = dataMatInConst[0]->getValue();

        if (!dataMatOutRootConst.empty())
            outroot = dataMatOutRootConst[0]->beginEdit();

        applyJT(out,in, outroot);

        dataMatOutConst[0]->endEdit();
        if (outroot != nullptr)
            dataMatOutRootConst[0]->endEdit();
    }

    const sofa::linearalgebra::BaseMatrix* getJ() override { return nullptr; }

    void draw(const core::visual::VisualParams* vparams) override;

    /**
    *	Stores al the articulation centers
    */
    std::vector< sofa::component::container::ArticulationCenter* > articulationCenters;

    container::ArticulatedHierarchyContainer* ahc;

private:
    core::State<In>* m_fromModel;
    core::State<Out>* m_toModel;
    core::State<InRoot>* m_fromRootModel;

    SingleLink<ArticulatedSystemMapping<TIn, TInRoot, TOut>,
                sofa::component::container::ArticulatedHierarchyContainer,
                BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK>            l_container;
    Data<sofa::Index> d_indexFromRoot; ///< Corresponding index if the base of the articulated system is attached to input2. Default is last index.

    sofa::type::Vec<1,sofa::type::Quat<SReal>> Buf_Rotation;
    std::vector< sofa::type::Vec<3,OutReal> > ArticulationAxis;
    std::vector< sofa::type::Vec<3,OutReal> > ArticulationPos;
    InVecCoord CoordinateBuf;
    InVecDeriv dxVec1Buf;
    OutVecDeriv dxRigidBuf;

    using core::Multi2Mapping<TIn, TInRoot, TOut>::d_componentState;

    void checkIndexFromRoot();
};

#if !defined(SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_CPP)

extern template class SOFA_ARTICULATEDSYSTEMPLUGIN_API ArticulatedSystemMapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types >;




#endif

} //namespace sofa::component::mapping
