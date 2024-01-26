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

#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/BaseMass.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
class CenterOfMassMultiMapping : public core::MultiMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CenterOfMassMultiMapping, TIn, TOut), SOFA_TEMPLATE2(core::MultiMapping, TIn, TOut));

    typedef core::MultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef In InDataTypes;
    typedef typename In::Coord    InCoord;
    typedef typename In::Deriv    InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;

    typedef Out OutDataTypes;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename OutCoord::value_type Real;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;


    typedef typename type::vector<OutVecCoord*> vecOutVecCoord;
    typedef typename type::vector<const InVecCoord*> vecConstInVecCoord;

    void apply(const core::MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos, const type::vector<const InDataVecCoord*>& dataVecInPos) override;
    //virtual void apply(const type::vector<OutVecCoord*>& outPos, const vecConstInVecCoord& inPos);

    void applyJ(const core::MechanicalParams* mparams, const type::vector<OutDataVecDeriv*>& dataVecOutVel, const type::vector<const InDataVecDeriv*>& dataVecInVel) override;
    //virtual void applyJ(const type::vector<OutVecDeriv*>& outDeriv, const type::vector<const  InVecDeriv*>& inDeriv);

    void applyJT(const core::MechanicalParams* mparams, const type::vector<InDataVecDeriv*>& dataVecOutForce, const type::vector<const OutDataVecDeriv*>& dataVecInForce) override;
    //virtual void applyJT(const type::vector< InVecDeriv*>& outDeriv, const type::vector<const OutVecDeriv*>& inDeriv);


    //virtual void apply(const vecOutVecCoord& outPos, const vecConstInVecCoord& inPos );
    //virtual void applyJ(const type::vector< OutVecDeriv*>& outDeriv, const type::vector<const InVecDeriv*>& inDeriv);
    //virtual void applyJT( const type::vector<InVecDeriv*>& outDeriv , const type::vector<const OutVecDeriv*>& inDeriv );

    void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override {}

    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;

protected:

    CenterOfMassMultiMapping()
        : Inherit()
    {
    }

    virtual ~CenterOfMassMultiMapping() {}

    type::vector<const core::behavior::BaseMass*> inputBaseMass;
    InVecCoord inputWeightedCOM;
    InVecDeriv inputWeightedForce;
    type::vector<double> inputTotalMass;
    double invTotalMass;
};

#if !defined(SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API CenterOfMassMultiMapping< defaulttype::Vec3Types, defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API CenterOfMassMultiMapping< defaulttype::Rigid3Types, defaulttype::Rigid3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API CenterOfMassMultiMapping< defaulttype::Rigid3Types, defaulttype::Vec3Types >;

#endif

} // namespace sofa::component::mapping::linear
