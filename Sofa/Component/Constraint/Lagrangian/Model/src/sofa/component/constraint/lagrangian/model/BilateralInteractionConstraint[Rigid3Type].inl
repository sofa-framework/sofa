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

#include <sofa/component/constraint/lagrangian/model/BilateralInteractionConstraint[Rigid3Type].h>
#include <sofa/component/constraint/lagrangian/model/BilateralInteractionConstraint.inl>

namespace sofa::component::constraint::lagrangian::model::bilateralinteractionconstraint
{

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralInteractionConstraint<Rigid3Types>::init(){
    unspecializedInit() ;
}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralInteractionConstraint<Rigid3Types>::bwdInit()
{
    if (!this->keepOrientDiff.getValue())
        return;

    helper::WriteAccessor<Data<VecDeriv > > wrest = this->restVector;

    if (wrest.size() > 0)
    {
        msg_warning() << "keepOrientationDifference is activated, rest_vector will be ignored! " ;
        wrest.resize(0);
    }

    const type::vector<int> &m1Indices = this->m1.getValue();
    const type::vector<int> &m2Indices = this->m2.getValue();

    unsigned minp = std::min(m1Indices.size(),m2Indices.size());

    const DataVecCoord &d_x1 = *this->mstate1->read(core::ConstVecCoordId::position());
    const DataVecCoord &d_x2 = *this->mstate2->read(core::ConstVecCoordId::position());

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();

    for (unsigned pid=0; pid<minp; pid++)
    {
        const Coord P = x1[m1Indices[pid]];
        const Coord Q = x2[m2Indices[pid]];

        type::Quat<SReal> qP, qQ, dQP;
        qP = P.getOrientation();
        qQ = Q.getOrientation();
        qP.normalize();
        qQ.normalize();
        dQP = qP.quatDiff(qQ, qP);
        dQP.normalize();

        Coord df;
        df.getCenter() = Q.getCenter() - P.getCenter();
        df.getOrientation() = dQP;
        this->initialDifference.push_back(df);
    }
}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralInteractionConstraint<Rigid3Types>::getConstraintResolution(const ConstraintParams* cParams,
                                                                           std::vector<ConstraintResolution*>& resTab,
                                                                           unsigned int& offset)
{
    SOFA_UNUSED(cParams);
    unsigned minp=std::min(this->m1.getValue().size(),
                           this->m2.getValue().size());
    for (unsigned pid=0; pid<minp; pid++)
    {
        resTab[offset] = new BilateralConstraintResolution3Dof();
        offset += 3;
        BilateralConstraintResolution3Dof* temp = new BilateralConstraintResolution3Dof();
        temp->setTolerance(d_numericalTolerance.getValue());	// specific (smaller) tolerance for the rotation
        resTab[offset] = temp;
        offset += 3;
    }
}

template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralInteractionConstraint<Rigid3Types>::buildConstraintMatrix(const ConstraintParams* cParams,
                                                                         DataMatrixDeriv &c1_d,
                                                                         DataMatrixDeriv &c2_d,
                                                                         unsigned int &constraintId,
                                                                         const DataVecCoord &x1, const DataVecCoord &x2)
{
    SOFA_UNUSED(cParams) ;
    const type::vector<int> &m1Indices = this->m1.getValue();
    const type::vector<int> &m2Indices = this->m2.getValue();

    unsigned minp = std::min(m1Indices.size(),m2Indices.size());
    this->cid.resize(minp);

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    const Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);
    const Vec<3, Real> vZero(0,0,0);

    for (unsigned pid=0; pid<minp; pid++)
    {
        int tm1 = m1Indices[pid];
        int tm2 = m2Indices[pid];

        this->cid[pid] = constraintId;
        constraintId += 6;

        //Apply constraint for position
        MatrixDerivRowIterator c1_it = c1.writeLine(this->cid[pid]);
        c1_it.addCol(tm1, Deriv(-cx, vZero));

        MatrixDerivRowIterator c2_it = c2.writeLine(this->cid[pid]);
        c2_it.addCol(tm2, Deriv(cx, vZero));

        c1_it = c1.writeLine(this->cid[pid] + 1);
        c1_it.setCol(tm1, Deriv(-cy, vZero));

        c2_it = c2.writeLine(this->cid[pid] + 1);
        c2_it.setCol(tm2, Deriv(cy, vZero));

        c1_it = c1.writeLine(this->cid[pid] + 2);
        c1_it.setCol(tm1, Deriv(-cz, vZero));

        c2_it = c2.writeLine(this->cid[pid] + 2);
        c2_it.setCol(tm2, Deriv(cz, vZero));

        //Apply constraint for orientation
        c1_it = c1.writeLine(this->cid[pid] + 3);
        c1_it.setCol(tm1, Deriv(vZero, -cx));

        c2_it = c2.writeLine(this->cid[pid] + 3);
        c2_it.setCol(tm2, Deriv(vZero, cx));

        c1_it = c1.writeLine(this->cid[pid] + 4);
        c1_it.setCol(tm1, Deriv(vZero, -cy));

        c2_it = c2.writeLine(this->cid[pid] + 4);
        c2_it.setCol(tm2, Deriv(vZero, cy));

        c1_it = c1.writeLine(this->cid[pid] + 5);
        c1_it.setCol(tm1, Deriv(vZero, -cz));

        c2_it = c2.writeLine(this->cid[pid] + 5);
        c2_it.setCol(tm2, Deriv(vZero, cz));
    }

    c1_d.endEdit();
    c2_d.endEdit();
}


template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralInteractionConstraint<Rigid3Types>::getConstraintViolation(const ConstraintParams* cParams,
                                                                          BaseVector *v,
                                                                          const DataVecCoord &d_x1, const DataVecCoord &d_x2,
                                                                          const DataVecDeriv &v1, const DataVecDeriv &v2)
{
    const type::vector<int> &m1Indices = this->m1.getValue();
    const type::vector<int> &m2Indices = this->m2.getValue();

    unsigned min = std::min(m1Indices.size(), m2Indices.size());
    const  VecDeriv& restVector = this->restVector.getValue();
    this->dfree.resize(min);

    const  VecCoord &x1 = d_x1.getValue();
    const  VecCoord &x2 = d_x2.getValue();

    for (unsigned pid=0; pid<min; pid++)
    {
        //Coord dof1 = x1[m1Indices[pid]];
        //Coord dof2 = x2[m2Indices[pid]];
        Coord dof1;

         if (this->keepOrientDiff.getValue()) {
             const Coord dof1c = x1[m1Indices[pid]];

             Coord corr=this->initialDifference[pid];
             type::Quat<SReal> df = corr.getOrientation();
             type::Quat<SReal> o1 = dof1c.getOrientation();
             type::Quat<SReal> ro1 = o1 * df;

             dof1.getCenter() = dof1c.getCenter() + corr.getCenter();
             dof1.getOrientation() = ro1;
         } else
             dof1 = x1[m1Indices[pid]];

        const Coord dof2 = x2[m2Indices[pid]];

        getVCenter(this->dfree[pid]) = dof2.getCenter() - dof1.getCenter();
        getVOrientation(this->dfree[pid]) =  dof1.rotate(this->q.angularDisplacement(dof2.getOrientation() ,
                                                                              dof1.getOrientation())); // angularDisplacement compute the rotation vector btw the two quaternions
        if (pid < restVector.size())
            this->dfree[pid] -= restVector[pid];

        for (unsigned int i=0 ; i<this->dfree[pid].size() ; i++)
            v->set(this->cid[pid]+i, this->dfree[pid][i]);
    }
}


template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralInteractionConstraint<Rigid3Types>::getVelocityViolation(BaseVector * /*v*/,
                                                                        const DataVecCoord &/*x1*/,
                                                                        const DataVecCoord &/*x2*/,
                                                                        const DataVecDeriv &/*v1*/,
                                                                        const DataVecDeriv &/*v2*/)
{

}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralInteractionConstraint<defaulttype::Rigid3Types>::addContact(Deriv norm,
                                                                           Coord P, Coord Q, Real contactDistance,
                                                                           int m1, int m2,
                                                                           Coord Pfree, Coord Qfree,
                                                                           long id, PersistentID localid)
{
    helper::WriteAccessor<Data<type::vector<int> > > wm1 = this->m1;
    helper::WriteAccessor<Data<type::vector<int> > > wm2 = this->m2;
    helper::WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.push_back(m1);
    wm2.push_back(m2);

    Deriv diff;
    getVCenter(diff) = Q.getCenter() - P.getCenter();
    getVOrientation(diff) =  P.rotate(this->q.angularDisplacement(Q.getOrientation() , P.getOrientation())) ; // angularDisplacement compute the rotation vector btw the two quaternions
    wrest.push_back(diff);
}

}
