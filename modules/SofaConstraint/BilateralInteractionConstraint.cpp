/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaConstraint/BilateralInteractionConstraint.inl>

#include <sofa/defaulttype/Vec3Types.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

#ifndef SOFA_FLOAT

template<>
void BilateralInteractionConstraint<Rigid3dTypes>::getConstraintResolution(const core::ConstraintParams* cParams, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    SOFA_UNUSED(cParams);
    unsigned minp=std::min(m1.getValue().size(),m2.getValue().size());
    for (unsigned pid=0; pid<minp; pid++)
    {
        // 	for(int i=0; i<6; i++)
        // 	resTab[offset++] = new BilateralConstraintResolution();

        resTab[offset] = new BilateralConstraintResolution3Dof();
        offset += 3;
        BilateralConstraintResolution3Dof* temp = new BilateralConstraintResolution3Dof();
        temp->tolerance = 0.01;	// specific (smaller) tolerance for the rotation
        resTab[offset] = temp;
        offset += 3;
    }
}

template <>
void BilateralInteractionConstraint<Rigid3dTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &constraintId
        , const DataVecCoord &/*x1*/, const DataVecCoord &/*x2*/)
{
    const helper::vector<int> &m1Indices = m1.getValue();
    const helper::vector<int> &m2Indices = m2.getValue();

    unsigned minp = std::min(m1Indices.size(),m2Indices.size());
    cid.resize(minp);

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    const Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);
    const Vec<3, Real> vZero(0,0,0);

    for (unsigned pid=0; pid<minp; pid++)
    {
        int tm1 = m1Indices[pid];
        int tm2 = m2Indices[pid];

        cid[pid] = constraintId;
        constraintId += 6;

        //Apply constraint for position
        MatrixDerivRowIterator c1_it = c1.writeLine(cid[pid]);
        c1_it.addCol(tm1, Deriv(-cx, vZero));

        MatrixDerivRowIterator c2_it = c2.writeLine(cid[pid]);
        c2_it.addCol(tm2, Deriv(cx, vZero));

        c1_it = c1.writeLine(cid[pid] + 1);
        c1_it.setCol(tm1, Deriv(-cy, vZero));

        c2_it = c2.writeLine(cid[pid] + 1);
        c2_it.setCol(tm2, Deriv(cy, vZero));

        c1_it = c1.writeLine(cid[pid] + 2);
        c1_it.setCol(tm1, Deriv(-cz, vZero));

        c2_it = c2.writeLine(cid[pid] + 2);
        c2_it.setCol(tm2, Deriv(cz, vZero));

        //Apply constraint for orientation
        c1_it = c1.writeLine(cid[pid] + 3);
        c1_it.setCol(tm1, Deriv(vZero, -cx));

        c2_it = c2.writeLine(cid[pid] + 3);
        c2_it.setCol(tm2, Deriv(vZero, cx));

        c1_it = c1.writeLine(cid[pid] + 4);
        c1_it.setCol(tm1, Deriv(vZero, -cy));

        c2_it = c2.writeLine(cid[pid] + 4);
        c2_it.setCol(tm2, Deriv(vZero, cy));

        c1_it = c1.writeLine(cid[pid] + 5);
        c1_it.setCol(tm1, Deriv(vZero, -cz));

        c2_it = c2.writeLine(cid[pid] + 5);
        c2_it.setCol(tm2, Deriv(vZero, cz));
    }

    c1_d.endEdit();
    c2_d.endEdit();
}


template <>
void BilateralInteractionConstraint<Rigid3dTypes>::getConstraintViolation(const core::ConstraintParams* /*cParams*/, defaulttype::BaseVector *v, const DataVecCoord &d_x1, const DataVecCoord &d_x2
        , const DataVecDeriv &/*v1*/, const DataVecDeriv &/*v2*/)
{
    const helper::vector<int> &m1Indices = m1.getValue();
    const helper::vector<int> &m2Indices = m2.getValue();

    unsigned minp = std::min(m1Indices.size(),m2Indices.size());

    const VecDeriv& restVector = this->restVector.getValue();
    dfree.resize(minp);

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();

    for (unsigned pid=0; pid<minp; pid++)
    {
        const Coord dof1 = x1[m1Indices[pid]];
        const Coord dof2 = x2[m2Indices[pid]];

        getVCenter(dfree[pid]) = dof2.getCenter() - dof1.getCenter();
        getVOrientation(dfree[pid]) =  dof1.rotate(q.angularDisplacement(dof2.getOrientation() , dof1.getOrientation())) ; // angularDisplacement compute the rotation vector btw the two quaternions

        if (pid < restVector.size())
            dfree[pid] -= restVector[pid];

        for (unsigned int i=0 ; i<dfree[pid].size() ; i++)
            v->set(cid[pid]+i, dfree[pid][i]);
    }
}


template <>
void BilateralInteractionConstraint<Rigid3dTypes>::getVelocityViolation(defaulttype::BaseVector * /*v*/, const DataVecCoord &/*x1*/, const DataVecCoord &/*x2*/, const DataVecDeriv &/*v1*/, const DataVecDeriv &/*v2*/)
{

}

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::addContact(Deriv /*norm*/, Coord P, Coord Q, Real /*contactDistance*/, int m1, int m2, Coord /*Pfree*/, Coord /*Qfree*/, long /*id*/, PersistentID /*localid*/)
{
    helper::WriteAccessor<Data<helper::vector<int> > > wm1 = this->m1;
    helper::WriteAccessor<Data<helper::vector<int> > > wm2 = this->m2;
    helper::WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.push_back(m1);
    wm2.push_back(m2);
    Deriv diff;
    getVCenter(diff) = Q.getCenter() - P.getCenter();
    getVOrientation(diff) =  P.rotate(q.angularDisplacement(Q.getOrientation() , P.getOrientation())) ; // angularDisplacement compute the rotation vector btw the two quaternions
    wrest.push_back(diff);
}

#endif

#ifndef SOFA_DOUBLE
template <>
void BilateralInteractionConstraint<Rigid3fTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &constraintId
        , const DataVecCoord &/*x1*/, const DataVecCoord &/*x2*/)
{
    const helper::vector<int> &m1Indices = m1.getValue();
    const helper::vector<int> &m2Indices = m2.getValue();

    unsigned minp = std::min(m1Indices.size(), m2Indices.size());
    cid.resize(minp);

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    const Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);
    const Vec<3, Real> vZero(0,0,0);

    for (unsigned pid=0; pid<minp; pid++)
    {
        int tm1 = m1Indices[pid];
        int tm2 = m2Indices[pid];

        cid[pid] = constraintId;
        constraintId += 6;

        //Apply constraint for position
        MatrixDerivRowIterator c1_it = c1.writeLine(cid[pid]);
        c1_it.addCol(tm1, Deriv(-cx, vZero));

        MatrixDerivRowIterator c2_it = c2.writeLine(cid[pid]);
        c2_it.addCol(tm2, Deriv(cx, vZero));

        c1_it = c1.writeLine(cid[pid] + 1);
        c1_it.setCol(tm1, Deriv(-cy, vZero));

        c2_it = c2.writeLine(cid[pid] + 1);
        c2_it.setCol(tm2, Deriv(cy, vZero));

        c1_it = c1.writeLine(cid[pid] + 2);
        c1_it.setCol(tm1, Deriv(-cz, vZero));

        c2_it = c2.writeLine(cid[pid] + 2);
        c2_it.setCol(tm2, Deriv(cz, vZero));

        //Apply constraint for orientation
        c1_it = c1.writeLine(cid[pid] + 3);
        c1_it.setCol(tm1, Deriv(vZero, -cx));

        c2_it = c2.writeLine(cid[pid] + 3);
        c2_it.setCol(tm2, Deriv(vZero, cx));

        c1_it = c1.writeLine(cid[pid] + 4);
        c1_it.setCol(tm1, Deriv(vZero, -cy));

        c2_it = c2.writeLine(cid[pid] + 4);
        c2_it.setCol(tm2, Deriv(vZero, cy));

        c1_it = c1.writeLine(cid[pid] + 5);
        c1_it.setCol(tm1, Deriv(vZero, -cz));

        c2_it = c2.writeLine(cid[pid] + 5);
        c2_it.setCol(tm2, Deriv(vZero, cz));
    }

    c1_d.endEdit();
    c2_d.endEdit();
}


template <>
void BilateralInteractionConstraint<Rigid3fTypes>::getConstraintViolation(const core::ConstraintParams* /*cParams*/, defaulttype::BaseVector *v, const DataVecCoord &d_x1, const DataVecCoord &d_x2
        , const DataVecDeriv &/*v1*/, const DataVecDeriv &/*v2*/)
{
    const helper::vector<int> &m1Indices = m1.getValue();
    const helper::vector<int> &m2Indices = m2.getValue();

    unsigned min = std::min(m1Indices.size(), m2Indices.size());
    const VecDeriv& restVector = this->restVector.getValue();
    dfree.resize(min);

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();

    for (unsigned pid=0; pid<min; pid++)
    {
        Coord dof1 = x1[m1Indices[pid]];
        Coord dof2 = x2[m2Indices[pid]];

        getVCenter(dfree[pid]) = dof2.getCenter() - dof1.getCenter();
        getVOrientation(dfree[pid]) =  dof1.rotate(q.angularDisplacement(dof2.getOrientation() , dof1.getOrientation())); // angularDisplacement compute the rotation vector btw the two quaternions
        if (pid < restVector.size())
            dfree[pid] -= restVector[pid];

        for (unsigned int i=0 ; i<dfree[pid].size() ; i++)
            v->set(cid[pid]+i, dfree[pid][i]);
    }
}

template <>
void BilateralInteractionConstraint<Rigid3fTypes>::getVelocityViolation(defaulttype::BaseVector * /*v*/, const DataVecCoord &/*x1*/, const DataVecCoord &/*x2*/, const DataVecDeriv &/*v1*/, const DataVecDeriv &/*v2*/)
{

}

template<>
void BilateralInteractionConstraint<Rigid3fTypes>::getConstraintResolution(const core::ConstraintParams* cParams, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    SOFA_UNUSED(cParams);
    unsigned minp=std::min(m1.getValue().size(),m2.getValue().size());
    for (unsigned pid=0; pid<minp; pid++)
    {
        // 	for(int i=0; i<6; i++)
        // 	resTab[offset++] = new BilateralConstraintResolution();

        resTab[offset] = new BilateralConstraintResolution3Dof();
        offset += 3;
        BilateralConstraintResolution3Dof* temp = new BilateralConstraintResolution3Dof();
        temp->tolerance = 0.0001;	// specific (smaller) tolerance for the rotation
        resTab[offset] = temp;
        offset += 3;
    }
}


template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::addContact(Deriv /*norm*/, Coord P, Coord Q, Real /*contactDistance*/, int m1, int m2, Coord /*Pfree*/, Coord /*Qfree*/, long /*id*/, PersistentID /*localid*/)
{
    helper::WriteAccessor<Data<helper::vector<int> > > wm1 = this->m1;
    helper::WriteAccessor<Data<helper::vector<int> > > wm2 = this->m2;
    helper::WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.push_back(m1);
    wm2.push_back(m2);
    Deriv diff;
    getVCenter(diff) = Q.getCenter() - P.getCenter();
    getVOrientation(diff) =  P.rotate(q.angularDisplacement(Q.getOrientation() , P.getOrientation())) ; // angularDisplacement compute the rotation vector btw the two quaternions
    wrest.push_back(diff);
}

#endif


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
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Vec3dTypes>;
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Vec3fTypes>;
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Rigid3fTypes>;
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

