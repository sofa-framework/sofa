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
#define SOFA_COMPONENT_CONSTRAINTSET_STOPPERLAGRANGIANCONSTRAINT_CPP
#include <sofa/component/constraint/lagrangian/model/FixedLagrangianConstraint.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/constraint/lagrangian/model/BilateralConstraintResolution.h>

namespace sofa::component::constraint::lagrangian::model
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


// Vec3D specialization
template<>
void FixedLagrangianConstraint<Vec3Types>::doBuildConstraintLine( MatrixDeriv &c, unsigned int &cIndex, unsigned int lineNumber)
{
    constexpr Coord cx(1,0,0), cy(0,1,0), cz(0,0,1);
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    cIndex += 3;

    MatrixDerivRowIterator c_it = c.writeLine(m_cid[lineNumber]);
    c_it.setCol(dofIdx, cx);

    c_it = c.writeLine(m_cid[lineNumber] + 1);
    c_it.setCol(dofIdx, cy);

    c_it = c.writeLine(m_cid[lineNumber] + 2);
    c_it.setCol(dofIdx, cz);

}

template<>
void FixedLagrangianConstraint<Vec3Types>::doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePose, const DataVecCoord * restPose,unsigned int lineNumber)
{
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    Coord dfree = freePose->getValue()[dofIdx] - restPose->getValue()[dofIdx];

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


// Rigid3D specialization
template<>
void FixedLagrangianConstraint<Rigid3Types>::doBuildConstraintLine( MatrixDeriv &c, unsigned int &cIndex, unsigned int lineNumber)
{
    constexpr type::Vec3 cx(1,0,0), cy(0,1,0), cz(0,0,1), zero(0,0,0);
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];

    cIndex += 6;

    MatrixDerivRowIterator c_it = c.writeLine(m_cid[lineNumber]);
    c_it.setCol(dofIdx, Deriv(cx, zero));

    c_it = c.writeLine(m_cid[lineNumber] + 1);
    c_it.setCol(dofIdx, Deriv(cy, zero));

    c_it = c.writeLine(m_cid[lineNumber] + 2);
    c_it.setCol(dofIdx, Deriv(cz, zero));

    c_it = c.writeLine(m_cid[lineNumber] + 3);
    c_it.setCol(dofIdx, Deriv(zero, cx));

    c_it = c.writeLine(m_cid[lineNumber] + 4);
    c_it.setCol(dofIdx, Deriv(zero, cy));

    c_it = c.writeLine(m_cid[lineNumber] + 5);
    c_it.setCol(dofIdx, Deriv(zero, cz));

}

template<>
void FixedLagrangianConstraint<Rigid3Types>::doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePose, const DataVecCoord * restPose,unsigned int lineNumber)
{
    unsigned dofIdx = d_fixAll.getValue() ? lineNumber : d_indices.getValue()[lineNumber];


    //TODO
    Deriv dfree;// = freePose->getValue()[dofIdx] - restPose->getValue()[dofIdx];

    resV->set(m_cid[lineNumber]  , dfree[0]);
    resV->set(m_cid[lineNumber]+1, dfree[1]);
    resV->set(m_cid[lineNumber]+2, dfree[2]);
    resV->set(m_cid[lineNumber]+3, dfree[3]);
    resV->set(m_cid[lineNumber]+4, dfree[4]);
    resV->set(m_cid[lineNumber]+5, dfree[5]);

}

template<>
void FixedLagrangianConstraint<Rigid3Types>::doGetSingleConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset, unsigned int lineNumber)
{
    resTab[offset] = new BilateralConstraintResolution3Dof(&getVCenter(m_prevForces[lineNumber]));
    offset += 3;
    resTab[offset] = new BilateralConstraintResolution3Dof(&getVOrientation(m_prevForces[lineNumber]));
    offset += 3;
}



int FixedLagrangianConstraintClass = core::RegisterObject("TODO-FixedLagrangianConstraint")
        .add< FixedLagrangianConstraint<Vec3Types> >()
        .add< FixedLagrangianConstraint<Rigid3Types> >();

template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint<Vec3Types>;


} //namespace sofa::component::constraint::lagrangian::model
