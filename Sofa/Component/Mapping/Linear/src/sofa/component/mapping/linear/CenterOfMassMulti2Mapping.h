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
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/VecId.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::mapping::linear
{

template <class TIn1, class TIn2, class TOut>
class CenterOfMassMulti2Mapping : public core::Multi2Mapping<TIn1, TIn2, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(CenterOfMassMulti2Mapping, TIn1, TIn2, TOut), SOFA_TEMPLATE3(core::Multi2Mapping, TIn1, TIn2, TOut));

    typedef core::Multi2Mapping<TIn1, TIn2, TOut> Inherit;
    typedef TIn1 In1;
    typedef TIn2 In2;
    typedef TOut Out;

    typedef In1 In1DataTypes;
    typedef typename In1::Coord    In1Coord;
    typedef typename In1::Deriv    In1Deriv;
    typedef typename In1::VecCoord In1VecCoord;
    typedef typename In1::VecDeriv In1VecDeriv;

    typedef In2 In2DataTypes;
    typedef typename In2::Coord    In2Coord;
    typedef typename In2::Deriv    In2Deriv;
    typedef typename In2::VecCoord In2VecCoord;
    typedef typename In2::VecDeriv In2VecDeriv;

    typedef Out OutDataTypes;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename OutCoord::value_type Real;

    typedef typename In1::MatrixDeriv In1MatrixDeriv;
    typedef Data<In1VecCoord> In1DataVecCoord;
    typedef Data<In1VecDeriv> In1DataVecDeriv;
    typedef Data<In1MatrixDeriv> In1DataMatrixDeriv;
    typedef typename In2::MatrixDeriv In2MatrixDeriv;
    typedef Data<In2VecCoord> In2DataVecCoord;
    typedef Data<In2VecDeriv> In2DataVecDeriv;
    typedef Data<In2MatrixDeriv> In2DataMatrixDeriv;

    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;


    void apply(
        const core::MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos,
        const type::vector<const In1DataVecCoord*>& dataVecIn1Pos ,
        const type::vector<const In2DataVecCoord*>& dataVecIn2Pos) override;
    void applyJ(
        const core::MechanicalParams* mparams, const type::vector< OutDataVecDeriv*>& dataVecOutVel,
        const type::vector<const In1DataVecDeriv*>& dataVecIn1Vel,
        const type::vector<const In2DataVecDeriv*>& dataVecIn2Vel) override;
    void applyJT(
        const core::MechanicalParams* mparams, const type::vector< In1DataVecDeriv*>& dataVecOut1Force,
        const type::vector< In2DataVecDeriv*>& dataVecOut2Force,
        const type::vector<const OutDataVecDeriv*>& dataVecInForce) override;
    void applyJT(
        const core::ConstraintParams* /*cparams*/, const type::vector< In1DataMatrixDeriv*>& /* dataMatOut1Const */ ,
        const type::vector< In2DataMatrixDeriv*>&  /*dataMatOut2Const*/ ,
        const type::vector<const OutDataMatrixDeriv*>& /*dataMatInConst*/) override
    {
        msg_warning() << "This object only support Direct Solving but an Indirect Solver in the scene is calling method applyJT(constraint) which is not implemented. This will produce un-expected behavior.";
    }

    void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override {}

    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;

protected:

    CenterOfMassMulti2Mapping()
    {
    }

    virtual ~CenterOfMassMulti2Mapping()
    {}

    type::vector<const core::behavior::BaseMass*> inputBaseMass1;
    type::vector<In1Coord> inputWeightedCOM1;
    type::vector<In1Deriv> inputWeightedForce1;
    type::vector<double> inputTotalMass1;

    type::vector<const core::behavior::BaseMass*> inputBaseMass2;
    type::vector<In2Coord> inputWeightedCOM2;
    type::vector<In2Deriv> inputWeightedForce2;
    type::vector<double> inputTotalMass2;

    double invTotalMass;
};

#if !defined(SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API CenterOfMassMulti2Mapping< defaulttype::Vec3Types, defaulttype::Rigid3Types, defaulttype::Vec3Types >;

#endif

} // namespace sofa::component::mapping::linear
