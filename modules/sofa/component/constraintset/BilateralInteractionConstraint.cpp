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
#define SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_CPP

#include <sofa/component/constraintset/BilateralInteractionConstraint.inl>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(BilateralInteractionConstraint)

int BilateralInteractionConstraintClass = core::RegisterObject("TODO-BilateralInteractionConstraint")
#ifndef SOFA_FLOAT
        .add< BilateralInteractionConstraint<Vec3dTypes> >()
        .add< BilateralInteractionConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< BilateralInteractionConstraint<Vec3fTypes> >()
        .add< BilateralInteractionConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT

#ifdef SOFA_DEV
template<>
void BilateralInteractionConstraint<Rigid3dTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
//	resTab[offset] = new BilateralConstraintResolution3Dof();
//	offset += 3;

    for(int i=0; i<6; i++)
        resTab[offset++] = new BilateralConstraintResolution();
}
#endif


template <>
void BilateralInteractionConstraint<Rigid3dTypes>::buildConstraintMatrix(unsigned int &constraintId, core::VecId)
{
    SparseVecDeriv svd1;
    SparseVecDeriv svd2;

    int tm1, tm2;
    tm1 = m1.getValue();
    tm2 = m2.getValue();

    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);
    Vec<3, Real> qId = q.toEulerVector();
    Vec<3, Real> vZero(0,0,0);


    cid = constraintId;
    constraintId+=6;

    //Apply constraint for position
    this->object1->setConstraintId(cid);
    svd1.add(tm1, Deriv(-cx, vZero));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid);
    svd2.add(tm2, Deriv(cx, vZero));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+1);
    svd1.set(tm1, Deriv(-cy, vZero));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+1);
    svd2.set(tm2, Deriv(cy, vZero));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+2);
    svd1.set(tm1, Deriv(-cz, vZero));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+2);
    svd2.set(tm2, Deriv(cz, vZero));
    c2.push_back(svd2);

    //Apply constraint for orientation
    this->object1->setConstraintId(cid+3);
    svd1.set(tm1, Deriv(vZero, -cx));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+3);
    svd2.set(tm2, Deriv(vZero, cx));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+4);
    svd1.set(tm1, Deriv(vZero, -cy));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+4);
    svd2.set(tm2, Deriv(vZero, cy));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+5);
    svd1.set(tm1, Deriv(vZero, -cz));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+5);
    svd2.set(tm2, Deriv(vZero, cz));
    c2.push_back(svd2);

}

template <>
void BilateralInteractionConstraint<Rigid3dTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion)
{
    if (!freeMotion)
        sout<<"WARNING has to be implemented for method based on non freeMotion"<<sendl;
    else
    {
        Coord dof1 = (*this->object1->getXfree())[m1.getValue()];
        Coord dof2 = (*this->object2->getXfree())[m2.getValue()];

        dfree.getVCenter() = dof2.getCenter() - dof1.getCenter();
        dfree.getVOrientation() =  dof1.rotate(q.angularDisplacement(dof2.getOrientation() , dof1.getOrientation())) ;
    }
    //std::cout << q.axisToQuat(Vec3f(0,0,1),-M_PI/2) << std::endl;

    for (unsigned int i=0 ; i<dfree.size() ; i++)
        v->set(cid+i, dfree[i]);
}




#endif

#ifndef SOFA_DOUBLE
template <>
void BilateralInteractionConstraint<Rigid3fTypes>::buildConstraintMatrix(unsigned int &constraintId, core::VecId)
{
    SparseVecDeriv svd1;
    SparseVecDeriv svd2;

    int tm1, tm2;
    tm1 = m1.getValue();
    tm2 = m2.getValue();

    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);
    Vec<3, Real> qId;
    Vec<3, Real> pId;

    cid = constraintId;
    constraintId+=6;

    //Apply constraint for position
    this->object1->setConstraintId(cid);
    svd1.add(tm1, Deriv(-cx, qId));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid);
    svd2.add(tm2, Deriv(cx, qId));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+1);
    svd1.set(tm1, Deriv(-cy, qId));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+1);
    svd2.set(tm2, Deriv(cy, qId));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+2);
    svd1.set(tm1, Deriv(-cz, qId));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+2);
    svd2.set(tm2, Deriv(cz, qId));
    c2.push_back(svd2);

    //Apply constraint for orientation
    this->object1->setConstraintId(cid+3);
    svd1.set(tm1, Deriv(pId, -cx));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+3);
    svd2.set(tm2, Deriv(pId, cx));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+4);
    svd1.set(tm1, Deriv(pId, -cy));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+4);
    svd2.set(tm2, Deriv(pId, cy));
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+5);
    svd1.set(tm1, Deriv(pId, -cz));
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+5);
    svd2.set(tm2, Deriv(pId, cz));
    c2.push_back(svd2);
}

template <>
void BilateralInteractionConstraint<Rigid3fTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion)
{
    if (!freeMotion)
        sout<<"WARNING has to be implemented for method based on non freeMotion"<<sendl;
    else
    {
        Coord dof1 = (*this->object1->getXfree())[m1.getValue()];
        Coord dof2 = (*this->object2->getXfree())[m1.getValue()];

        dfree.getVCenter() = dof2.getCenter() - dof1.getCenter();
        dfree.getVOrientation() =  dof1.rotate(q.angularDisplacement(dof2.getOrientation() , dof1.getOrientation())) ;
    }

    for (unsigned int i=0 ; i<dfree.size() ; i++)
        v->set(cid+i, dfree[i]);
}

#ifdef SOFA_DEV
template<>
void BilateralInteractionConstraint<Rigid3fTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
//	resTab[offset] = new BilateralConstraintResolution3Dof();
//	offset += 3;

    for(int i=0; i<6; i++)
        resTab[offset++] = new BilateralConstraintResolution();
}
#endif

#endif





#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint<Vec3dTypes>;
template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint<Vec3fTypes>;
template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint<Rigid3fTypes>;
#endif




} // namespace constraintset

} // namespace component

} // namespace sofa

