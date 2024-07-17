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
#define SOFA_COMPONENT_CONSTRAINTSET_FIXEDLAGRANGIANCONSTRAINT_CPP
#include <sofa/component/constraint/lagrangian/model/FixedLagrangianConstraint.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/constraint/lagrangian/model/BilateralConstraintResolution.h>

namespace sofa::component::constraint::lagrangian::model
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

// Vec6 specialization
template <>
void FixedLagrangianConstraint< Vec6Types >::doBuildConstraintLine( helper::WriteAccessor<DataMatrixDeriv> &c, unsigned int lineNumber)
{
    constexpr Coord c0(1,0,0,0,0,0), c1(0,1,0,0,0,0), c2(0,0,1,0,0,0), c3(0,0,0,1,0,0), c4(0,0,0,0,1,0), c5(0,0,0,0,0,1);
    const unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    MatrixDerivRowIterator c_it = c->writeLine(m_cid[lineNumber]);
    c_it.setCol(dofIdx, c0);

    c_it = c->writeLine(m_cid[lineNumber] + 1);
    c_it.setCol(dofIdx, c1);

    c_it = c->writeLine(m_cid[lineNumber] + 2);
    c_it.setCol(dofIdx, c2);

    c_it = c->writeLine(m_cid[lineNumber] + 3);
    c_it.setCol(dofIdx, c3);

    c_it = c->writeLine(m_cid[lineNumber] + 4);
    c_it.setCol(dofIdx, c4);

    c_it = c->writeLine(m_cid[lineNumber] + 5);
    c_it.setCol(dofIdx, c5);


}

template<>
void FixedLagrangianConstraint<Vec6Types>::doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePos, const DataVecCoord * restPos,unsigned int lineNumber)
{
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];
    const Coord dfree = freePos->getValue()[dofIdx] - restPos->getValue()[dofIdx];

    resV->set(m_cid[lineNumber]  , dfree[0]);
    resV->set(m_cid[lineNumber]+1, dfree[1]);
    resV->set(m_cid[lineNumber]+2, dfree[2]);
    resV->set(m_cid[lineNumber]+3, dfree[3]);
    resV->set(m_cid[lineNumber]+4, dfree[4]);
    resV->set(m_cid[lineNumber]+5, dfree[5]);

}

template<>
void FixedLagrangianConstraint<Vec6Types>::doGetSingleConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset, unsigned int lineNumber)
{
    SOFA_UNUSED(lineNumber);
    resTab[offset] = new BilateralConstraintResolutionNDof(6);
    offset += 6;
}

// Vec3 specialization
template <>
void FixedLagrangianConstraint< Vec3Types >::doBuildConstraintLine( helper::WriteAccessor<DataMatrixDeriv> &c, unsigned int lineNumber)
{
    constexpr Coord cx(1,0,0), cy(0,1,0), cz(0,0,1);
    const unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    MatrixDerivRowIterator c_it = c->writeLine(m_cid[lineNumber]);
    c_it.setCol(dofIdx, cx);

    c_it = c->writeLine(m_cid[lineNumber] + 1);
    c_it.setCol(dofIdx, cy);

    c_it = c->writeLine(m_cid[lineNumber] + 2);
    c_it.setCol(dofIdx, cz);

}

template<>
void FixedLagrangianConstraint<Vec3Types>::doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePos, const DataVecCoord * restPos,unsigned int lineNumber)
{
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];
    const Coord dfree = freePos->getValue()[dofIdx] - restPos->getValue()[dofIdx];

    resV->set(m_cid[lineNumber]  , dfree[0]);
    resV->set(m_cid[lineNumber]+1, dfree[1]);
    resV->set(m_cid[lineNumber]+2, dfree[2]);

}

template<>
void FixedLagrangianConstraint<Vec3Types>::doGetSingleConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset, unsigned int lineNumber)
{
    resTab[offset] = new BilateralConstraintResolution3Dof(&m_prevForces[lineNumber]);
    offset += 3;
}


// Vec2 specialization
template <>
void FixedLagrangianConstraint< Vec2Types >::doBuildConstraintLine( helper::WriteAccessor<DataMatrixDeriv> &c, unsigned int lineNumber)
{
    constexpr Coord cx(1, 0), cy(0, 1);
    const unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    MatrixDerivRowIterator c_it = c->writeLine(m_cid[lineNumber]);
    c_it.setCol(dofIdx, cx);

    c_it = c->writeLine(m_cid[lineNumber] + 1);
    c_it.setCol(dofIdx, cy);
}

template<>
void FixedLagrangianConstraint<Vec2Types>::doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePos, const DataVecCoord * restPos,unsigned int lineNumber)
{
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];
    const Coord dfree = freePos->getValue()[dofIdx] - restPos->getValue()[dofIdx];

    resV->set(m_cid[lineNumber]  , dfree[0]);
    resV->set(m_cid[lineNumber]+1, dfree[1]);

}

template<>
void FixedLagrangianConstraint<Vec2Types>::doGetSingleConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset, unsigned int lineNumber)
{
    SOFA_UNUSED(lineNumber);
    resTab[offset] = new BilateralConstraintResolutionNDof(2);
    offset += 2;
}

// Vec1 specialization
template <>
void FixedLagrangianConstraint< Vec1Types >::doBuildConstraintLine( helper::WriteAccessor<DataMatrixDeriv> &c, unsigned int lineNumber)
{
    constexpr Coord cx(1);
    const unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    MatrixDerivRowIterator c_it = c->writeLine(m_cid[lineNumber]);
    c_it.setCol(dofIdx, cx);
}

template<>
void FixedLagrangianConstraint<Vec1Types>::doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePos, const DataVecCoord * restPos,unsigned int lineNumber)
{
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];
    const Coord dfree = freePos->getValue()[dofIdx] - restPos->getValue()[dofIdx];

    resV->set(m_cid[lineNumber]  , dfree[0]);
}

template<>
void FixedLagrangianConstraint<Vec1Types>::doGetSingleConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset, unsigned int lineNumber)
{
    SOFA_UNUSED(lineNumber);
    resTab[offset] = new BilateralConstraintResolutionNDof(1);
    offset += 1;
}

// Rigid3 specialization
template<>
void FixedLagrangianConstraint<Rigid3Types>::doBuildConstraintLine( helper::WriteAccessor<DataMatrixDeriv> &c, unsigned int lineNumber)
{
    constexpr type::Vec3 cx(1,0,0), cy(0,1,0), cz(0,0,1), zero(0,0,0);
    const unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    MatrixDerivRowIterator c_it = c->writeLine(m_cid[lineNumber]);
    c_it.setCol(dofIdx, Deriv(cx, zero));

    c_it = c->writeLine(m_cid[lineNumber] + 1);
    c_it.setCol(dofIdx, Deriv(cy, zero));

    c_it = c->writeLine(m_cid[lineNumber] + 2);
    c_it.setCol(dofIdx, Deriv(cz, zero));

    c_it = c->writeLine(m_cid[lineNumber] + 3);
    c_it.setCol(dofIdx, Deriv(zero, cx));

    c_it = c->writeLine(m_cid[lineNumber] + 4);
    c_it.setCol(dofIdx, Deriv(zero, cy));

    c_it = c->writeLine(m_cid[lineNumber] + 5);
    c_it.setCol(dofIdx, Deriv(zero, cz));

}

template<>
void FixedLagrangianConstraint<Rigid3Types>::doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePose, const DataVecCoord * restPose,unsigned int lineNumber)
{
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];


    const sofa::type::Vec3 pfree = (freePose->getValue()[dofIdx].getCenter()  - restPose->getValue()[dofIdx].getCenter());
    const sofa::type::Vec3 ofree =  ( freePose->getValue()[dofIdx].getOrientation()*restPose->getValue()[dofIdx].getOrientation().inverse()).toEulerVector();


    resV->set(m_cid[lineNumber]  , pfree[0]);
    resV->set(m_cid[lineNumber]+1, pfree[1]);
    resV->set(m_cid[lineNumber]+2, pfree[2]);
    resV->set(m_cid[lineNumber]+3, ofree[0]);
    resV->set(m_cid[lineNumber]+4, ofree[1]);
    resV->set(m_cid[lineNumber]+5, ofree[2]);

}

template<>
void FixedLagrangianConstraint<Rigid3Types>::doGetSingleConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset, unsigned int lineNumber)
{
    resTab[offset] = new BilateralConstraintResolution3Dof(&getVCenter(m_prevForces[lineNumber]));
    offset += 3;
    resTab[offset] = new BilateralConstraintResolution3Dof(&getVOrientation(m_prevForces[lineNumber]));
    offset += 3;
}



int FixedLagrangianConstraintClass = core::RegisterObject("Lagrangian-based fixation of DOFs of the model")
        .add< FixedLagrangianConstraint<Vec6Types> >()
        .add< FixedLagrangianConstraint<Vec3Types> >()
        .add< FixedLagrangianConstraint<Vec2Types> >()
        .add< FixedLagrangianConstraint<Vec1Types> >()
        .add< FixedLagrangianConstraint<Rigid3Types> >();

template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint<Vec6Types>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint<Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint<Vec2Types>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint<Vec1Types>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint<Rigid3Types>;


} //namespace sofa::component::constraint::lagrangian::model
