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
#define SOFA_COMPONENT_CONSTRAINTSET_BILATERALLAGRANGIANCONSTRAINT_CPP

#include <sofa/component/constraint/lagrangian/model/BilateralLagrangianConstraint.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::constraint::lagrangian::model
{

class RigidImpl {};


template<>
class BilateralLagrangianConstraintSpecialization<RigidImpl>
{
public:

    template<class T>
    static void bwdInit(BilateralLagrangianConstraint<T>& self) {
        if (!self.keepOrientDiff.getValue())
            return;

        helper::WriteAccessor<Data<typename BilateralLagrangianConstraint<T>::VecDeriv > > wrest = self.restVector;

        if (wrest.size() > 0) {
            msg_warning("BilateralLagrangianConstraintSpecialization") << "keepOrientationDifference is activated, rest_vector will be ignored! " ;
            wrest.resize(0);
        }

        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m1Indices = self.m1.getValue();
        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m2Indices = self.m2.getValue();

        const unsigned minp = std::min(m1Indices.size(),m2Indices.size());

        const typename BilateralLagrangianConstraint<T>::DataVecCoord &d_x1 = *self.mstate1->read(core::ConstVecCoordId::position());
        const typename BilateralLagrangianConstraint<T>::DataVecCoord &d_x2 = *self.mstate2->read(core::ConstVecCoordId::position());

        const typename BilateralLagrangianConstraint<T>::VecCoord &x1 = d_x1.getValue();
        const typename BilateralLagrangianConstraint<T>::VecCoord &x2 = d_x2.getValue();

        for (unsigned pid=0; pid<minp; pid++)
        {
            const typename BilateralLagrangianConstraint<T>::Coord P = x1[m1Indices[pid]];
            const typename BilateralLagrangianConstraint<T>::Coord Q = x2[m2Indices[pid]];

            type::Quat<SReal> qP, qQ, dQP;
            qP = P.getOrientation();
            qQ = Q.getOrientation();
            qP.normalize();
            qQ.normalize();
            dQP = qP.quatDiff(qQ, qP);
            dQP.normalize();

            typename BilateralLagrangianConstraint<T>::Coord df;
            df.getCenter() = Q.getCenter() - P.getCenter();
            df.getOrientation() = dQP;
            self.initialDifference.push_back(df);

        }
    }


    template<class T>
    static void getConstraintResolution(BilateralLagrangianConstraint<T>& self,
                                        const ConstraintParams* cParams,
                                        std::vector<ConstraintResolution*>& resTab,
                                        unsigned int& offset, double tolerance)
    {
        SOFA_UNUSED(cParams);
        const unsigned minp=std::min(self.m1.getValue().size(),
                                     self.m2.getValue().size());
        for (unsigned pid=0; pid<minp; pid++)
        {
            resTab[offset] = new BilateralConstraintResolution3Dof();
            offset += 3;
            BilateralConstraintResolution3Dof* temp = new BilateralConstraintResolution3Dof();
            temp->setTolerance(tolerance);	// specific (smaller) tolerance for the rotation
            resTab[offset] = temp;
            offset += 3;
        }
    }


    template <class T>
    static void buildConstraintMatrix(BilateralLagrangianConstraint<T>& self,
                                      const ConstraintParams* cParams,
                                      typename BilateralLagrangianConstraint<T>::DataMatrixDeriv &c1_d,
                                      typename BilateralLagrangianConstraint<T>::DataMatrixDeriv &c2_d,
                                      unsigned int &constraintId,
                                      const typename BilateralLagrangianConstraint<T>::DataVecCoord &/*x1*/,
                                      const typename BilateralLagrangianConstraint<T>::DataVecCoord &/*x2*/)
    {
        SOFA_UNUSED(cParams) ;
        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m1Indices = self.m1.getValue();
        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m2Indices = self.m2.getValue();

        unsigned minp = std::min(m1Indices.size(),m2Indices.size());
        self.cid.resize(minp);

        typename BilateralLagrangianConstraint<T>::MatrixDeriv &c1 = *c1_d.beginEdit();
        typename BilateralLagrangianConstraint<T>::MatrixDeriv &c2 = *c2_d.beginEdit();

        const Vec<3, typename BilateralLagrangianConstraint<T>::Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);
        const Vec<3, typename BilateralLagrangianConstraint<T>::Real> vZero(0,0,0);

        for (unsigned pid=0; pid<minp; pid++)
        {
            int tm1 = m1Indices[pid];
            int tm2 = m2Indices[pid];

            self.cid[pid] = constraintId;
            constraintId += 6;

            //Apply constraint for position
            typename BilateralLagrangianConstraint<T>::MatrixDerivRowIterator c1_it = c1.writeLine(self.cid[pid]);
            c1_it.addCol(tm1, typename BilateralLagrangianConstraint<T>::Deriv(-cx, vZero));

            typename BilateralLagrangianConstraint<T>::MatrixDerivRowIterator c2_it = c2.writeLine(self.cid[pid]);
            c2_it.addCol(tm2, typename BilateralLagrangianConstraint<T>::Deriv(cx, vZero));

            c1_it = c1.writeLine(self.cid[pid] + 1);
            c1_it.setCol(tm1, typename BilateralLagrangianConstraint<T>::Deriv(-cy, vZero));

            c2_it = c2.writeLine(self.cid[pid] + 1);
            c2_it.setCol(tm2, typename BilateralLagrangianConstraint<T>::Deriv(cy, vZero));

            c1_it = c1.writeLine(self.cid[pid] + 2);
            c1_it.setCol(tm1, typename BilateralLagrangianConstraint<T>::Deriv(-cz, vZero));

            c2_it = c2.writeLine(self.cid[pid] + 2);
            c2_it.setCol(tm2, typename BilateralLagrangianConstraint<T>::Deriv(cz, vZero));

            //Apply constraint for orientation
            c1_it = c1.writeLine(self.cid[pid] + 3);
            c1_it.setCol(tm1, typename BilateralLagrangianConstraint<T>::Deriv(vZero, -cx));

            c2_it = c2.writeLine(self.cid[pid] + 3);
            c2_it.setCol(tm2, typename BilateralLagrangianConstraint<T>::Deriv(vZero, cx));

            c1_it = c1.writeLine(self.cid[pid] + 4);
            c1_it.setCol(tm1, typename BilateralLagrangianConstraint<T>::Deriv(vZero, -cy));

            c2_it = c2.writeLine(self.cid[pid] + 4);
            c2_it.setCol(tm2, typename BilateralLagrangianConstraint<T>::Deriv(vZero, cy));

            c1_it = c1.writeLine(self.cid[pid] + 5);
            c1_it.setCol(tm1, typename BilateralLagrangianConstraint<T>::Deriv(vZero, -cz));

            c2_it = c2.writeLine(self.cid[pid] + 5);
            c2_it.setCol(tm2, typename BilateralLagrangianConstraint<T>::Deriv(vZero, cz));
        }

        c1_d.endEdit();
        c2_d.endEdit();
    }


    template <class T>
    static void getConstraintViolation(BilateralLagrangianConstraint<T>& self,
                                const ConstraintParams* /*cParams*/,
                                BaseVector *v,
                                const  typename BilateralLagrangianConstraint<T>::DataVecCoord &d_x1,
                                const  typename BilateralLagrangianConstraint<T>::DataVecCoord &d_x2,
                                const  typename BilateralLagrangianConstraint<T>::DataVecDeriv &/*v1*/,
                                const  typename BilateralLagrangianConstraint<T>::DataVecDeriv &/*v2*/)
    {
        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m1Indices = self.m1.getValue();
        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m2Indices = self.m2.getValue();

        unsigned min = std::min(m1Indices.size(), m2Indices.size());
        const  typename BilateralLagrangianConstraint<T>::VecDeriv& restVector = self.restVector.getValue();
        self.dfree.resize(min);

        const  typename BilateralLagrangianConstraint<T>::VecCoord &x1 = d_x1.getValue();
        const  typename BilateralLagrangianConstraint<T>::VecCoord &x2 = d_x2.getValue();

        for (unsigned pid=0; pid<min; pid++)
        {
            //typename BilateralLagrangianConstraint<T>::Coord dof1 = x1[m1Indices[pid]];
            //typename BilateralLagrangianConstraint<T>::Coord dof2 = x2[m2Indices[pid]];
            typename BilateralLagrangianConstraint<T>::Coord dof1;

             if (self.keepOrientDiff.getValue()) {
                 const typename BilateralLagrangianConstraint<T>::Coord dof1c = x1[m1Indices[pid]];

                 typename BilateralLagrangianConstraint<T>::Coord corr=self.initialDifference[pid];
                 type::Quat<SReal> df = corr.getOrientation();
                 type::Quat<SReal> o1 = dof1c.getOrientation();
                 type::Quat<SReal> ro1 = o1 * df;

                 dof1.getCenter() = dof1c.getCenter() + corr.getCenter();
                 dof1.getOrientation() = ro1;
             } else
                 dof1 = x1[m1Indices[pid]];

            const typename BilateralLagrangianConstraint<T>::Coord dof2 = x2[m2Indices[pid]];

            getVCenter(self.dfree[pid]) = dof2.getCenter() - dof1.getCenter();
            getVOrientation(self.dfree[pid]) =  dof1.rotate(self.q.angularDisplacement(dof2.getOrientation() ,
                                                                                  dof1.getOrientation())); // angularDisplacement compute the rotation vector btw the two quaternions
            if (pid < restVector.size())
                self.dfree[pid] -= restVector[pid];

            for (unsigned int i=0 ; i<self.dfree[pid].size() ; i++)
                v->set(self.cid[pid]+i, self.dfree[pid][i]);
        }
    }


    template <class T, typename MyClass = BilateralLagrangianConstraint<T> >
    static void addContact(BilateralLagrangianConstraint<T>& self, typename MyClass::Deriv /*norm*/,
                           typename MyClass::Coord P, typename MyClass::Coord Q,
                           typename MyClass::Real /*contactDistance*/, int m1, int m2,
                           typename MyClass::Coord /*Pfree*/, typename MyClass::Coord /*Qfree*/, long /*id*/, typename MyClass::PersistentID /*localid*/)
    {
        helper::WriteAccessor<Data<typename BilateralLagrangianConstraint<T>::SubsetIndices > > wm1 = self.m1;
        helper::WriteAccessor<Data<typename BilateralLagrangianConstraint<T>::SubsetIndices > > wm2 = self.m2;
        helper::WriteAccessor<Data<typename MyClass::VecDeriv > > wrest = self.restVector;
        wm1.push_back(m1);
        wm2.push_back(m2);

        typename MyClass::Deriv diff;
        getVCenter(diff) = Q.getCenter() - P.getCenter();
        getVOrientation(diff) =  P.rotate(self.q.angularDisplacement(Q.getOrientation() , P.getOrientation())) ; // angularDisplacement compute the rotation vector btw the two quaternions
        wrest.push_back(diff);
    }

};


template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::init(){
    unspecializedInit() ;
}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::bwdInit() {
    BilateralLagrangianConstraintSpecialization<RigidImpl>::bwdInit(*this);
}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::getConstraintResolution(const ConstraintParams* cParams,
                                                                           std::vector<ConstraintResolution*>& resTab,
                                                                           unsigned int& offset)
{
    BilateralLagrangianConstraintSpecialization<RigidImpl>::getConstraintResolution(*this,
                                                                                     cParams, resTab, offset,
                                                                                     d_numericalTolerance.getValue()) ;
}

template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::buildConstraintMatrix(const ConstraintParams* cParams,
                                                                         DataMatrixDeriv &c1_d,
                                                                         DataMatrixDeriv &c2_d,
                                                                         unsigned int &constraintId,
                                                                         const DataVecCoord &x1, const DataVecCoord &x2)
{
    BilateralLagrangianConstraintSpecialization<RigidImpl>::buildConstraintMatrix(*this,
                                                                                   cParams, c1_d, c2_d, constraintId,
                                                                                   x1, x2) ;
}


template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::getConstraintViolation(const ConstraintParams* cParams,
                                                                          BaseVector *v,
                                                                          const DataVecCoord &d_x1, const DataVecCoord &d_x2,
                                                                          const DataVecDeriv &v1, const DataVecDeriv &v2)
{
    BilateralLagrangianConstraintSpecialization<RigidImpl>::getConstraintViolation(*this,
                                                                                    cParams, v, d_x1, d_x2,
                                                                                    v1, v2) ;
}


template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::getVelocityViolation(BaseVector * /*v*/,
                                                                        const DataVecCoord &/*x1*/,
                                                                        const DataVecCoord &/*x2*/,
                                                                        const DataVecDeriv &/*v1*/,
                                                                        const DataVecDeriv &/*v2*/)
{

}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<defaulttype::Rigid3Types>::addContact(Deriv norm,
                                                                           Coord P, Coord Q, Real contactDistance,
                                                                           int m1, int m2,
                                                                           Coord Pfree, Coord Qfree,
                                                                           long id, PersistentID localid)
{
    BilateralLagrangianConstraintSpecialization<RigidImpl>::addContact(*this,
                                                                        norm, P, Q, contactDistance, m1, m2, Pfree, Qfree,
                                                                        id, localid) ;
}



int BilateralLagrangianConstraintClass = core::RegisterObject("BilateralLagrangianConstraint defining an holonomic equality constraint (attachment)")
        .add< BilateralLagrangianConstraint<Vec3Types> >()
        .add< BilateralLagrangianConstraint<Rigid3Types> >()
        ;

template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralLagrangianConstraint<Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralLagrangianConstraint<Rigid3Types>;

} //namespace sofa::component::constraint::lagrangian::model
