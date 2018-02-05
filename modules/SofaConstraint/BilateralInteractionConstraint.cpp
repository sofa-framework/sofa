/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

namespace bilateralinteractionconstraint
{


class RigidImpl {};


template<>
class BilateralInteractionConstraintSpecialization<RigidImpl>
{
public:

    template<class T>
    static void bwdInit(BilateralInteractionConstraint<T>& self) {
        if (!self.keepOrientDiff.getValue())
            return;

        helper::WriteAccessor<Data<typename BilateralInteractionConstraint<T>::VecDeriv > > wrest = self.restVector;

        if (wrest.size() > 0) {
            msg_warning("BilateralInteractionConstraintSpecialization") << "keepOrientationDifference is activated, rest_vector will be ignored! " ;
            wrest.resize(0);
        }

        const helper::vector<int> &m1Indices = self.m1.getValue();
        const helper::vector<int> &m2Indices = self.m2.getValue();

        unsigned minp = std::min(m1Indices.size(),m2Indices.size());

        const typename BilateralInteractionConstraint<T>::DataVecCoord &d_x1 = *self.mstate1->read(core::ConstVecCoordId::position());
        const typename BilateralInteractionConstraint<T>::DataVecCoord &d_x2 = *self.mstate2->read(core::ConstVecCoordId::position());

        const typename BilateralInteractionConstraint<T>::VecCoord &x1 = d_x1.getValue();
        const typename BilateralInteractionConstraint<T>::VecCoord &x2 = d_x2.getValue();

        for (unsigned pid=0; pid<minp; pid++)
        {
            const typename BilateralInteractionConstraint<T>::Coord P = x1[m1Indices[pid]];
            const typename BilateralInteractionConstraint<T>::Coord Q = x2[m2Indices[pid]];

            defaulttype::Quat qP, qQ, dQP, qQ2;
            qP = P.getOrientation();
            qQ = Q.getOrientation();
            qP.normalize();
            qQ.normalize();
            dQP = qDiff(qQ, qP);
            dQP.normalize();

            qQ2 = qP*dQP;

            typename BilateralInteractionConstraint<T>::Coord df;
            df.getCenter() = Q.getCenter() - P.getCenter();
            df.getOrientation() = dQP;
            self.initialDifference.push_back(df);

        }
    }


    template<class T>
    static void getConstraintResolution(BilateralInteractionConstraint<T>& self,
                                        const ConstraintParams* cParams,
                                        std::vector<ConstraintResolution*>& resTab,
                                        unsigned int& offset, double tolerance)
    {
        SOFA_UNUSED(cParams);
        unsigned minp=std::min(self.m1.getValue().size(),
                               self.m2.getValue().size());
        for (unsigned pid=0; pid<minp; pid++)
        {
            resTab[offset] = new BilateralConstraintResolution3Dof();
            offset += 3;
            BilateralConstraintResolution3Dof* temp = new BilateralConstraintResolution3Dof();
            temp->tolerance = tolerance;	// specific (smaller) tolerance for the rotation
            resTab[offset] = temp;
            offset += 3;
        }
    }


    template <class T>
    static void buildConstraintMatrix(BilateralInteractionConstraint<T>& self,
                                      const ConstraintParams* cParams,
                                      typename BilateralInteractionConstraint<T>::DataMatrixDeriv &c1_d,
                                      typename BilateralInteractionConstraint<T>::DataMatrixDeriv &c2_d,
                                      unsigned int &constraintId,
                                      const typename BilateralInteractionConstraint<T>::DataVecCoord &/*x1*/,
                                      const typename BilateralInteractionConstraint<T>::DataVecCoord &/*x2*/)
    {
        SOFA_UNUSED(cParams) ;
        const helper::vector<int> &m1Indices = self.m1.getValue();
        const helper::vector<int> &m2Indices = self.m2.getValue();

        unsigned minp = std::min(m1Indices.size(),m2Indices.size());
        self.cid.resize(minp);

        typename BilateralInteractionConstraint<T>::MatrixDeriv &c1 = *c1_d.beginEdit();
        typename BilateralInteractionConstraint<T>::MatrixDeriv &c2 = *c2_d.beginEdit();

        const Vec<3, typename BilateralInteractionConstraint<T>::Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);
        const Vec<3, typename BilateralInteractionConstraint<T>::Real> vZero(0,0,0);

        for (unsigned pid=0; pid<minp; pid++)
        {
            int tm1 = m1Indices[pid];
            int tm2 = m2Indices[pid];

            self.cid[pid] = constraintId;
            constraintId += 6;

            //Apply constraint for position
            typename BilateralInteractionConstraint<T>::MatrixDerivRowIterator c1_it = c1.writeLine(self.cid[pid]);
            c1_it.addCol(tm1, typename BilateralInteractionConstraint<T>::Deriv(-cx, vZero));

            typename BilateralInteractionConstraint<T>::MatrixDerivRowIterator c2_it = c2.writeLine(self.cid[pid]);
            c2_it.addCol(tm2, typename BilateralInteractionConstraint<T>::Deriv(cx, vZero));

            c1_it = c1.writeLine(self.cid[pid] + 1);
            c1_it.setCol(tm1, typename BilateralInteractionConstraint<T>::Deriv(-cy, vZero));

            c2_it = c2.writeLine(self.cid[pid] + 1);
            c2_it.setCol(tm2, typename BilateralInteractionConstraint<T>::Deriv(cy, vZero));

            c1_it = c1.writeLine(self.cid[pid] + 2);
            c1_it.setCol(tm1, typename BilateralInteractionConstraint<T>::Deriv(-cz, vZero));

            c2_it = c2.writeLine(self.cid[pid] + 2);
            c2_it.setCol(tm2, typename BilateralInteractionConstraint<T>::Deriv(cz, vZero));

            //Apply constraint for orientation
            c1_it = c1.writeLine(self.cid[pid] + 3);
            c1_it.setCol(tm1, typename BilateralInteractionConstraint<T>::Deriv(vZero, -cx));

            c2_it = c2.writeLine(self.cid[pid] + 3);
            c2_it.setCol(tm2, typename BilateralInteractionConstraint<T>::Deriv(vZero, cx));

            c1_it = c1.writeLine(self.cid[pid] + 4);
            c1_it.setCol(tm1, typename BilateralInteractionConstraint<T>::Deriv(vZero, -cy));

            c2_it = c2.writeLine(self.cid[pid] + 4);
            c2_it.setCol(tm2, typename BilateralInteractionConstraint<T>::Deriv(vZero, cy));

            c1_it = c1.writeLine(self.cid[pid] + 5);
            c1_it.setCol(tm1, typename BilateralInteractionConstraint<T>::Deriv(vZero, -cz));

            c2_it = c2.writeLine(self.cid[pid] + 5);
            c2_it.setCol(tm2, typename BilateralInteractionConstraint<T>::Deriv(vZero, cz));
        }

        c1_d.endEdit();
        c2_d.endEdit();
    }


    template <class T>
    static void getConstraintViolation(BilateralInteractionConstraint<T>& self,
                                const ConstraintParams* /*cParams*/,
                                BaseVector *v,
                                const  typename BilateralInteractionConstraint<T>::DataVecCoord &d_x1,
                                const  typename BilateralInteractionConstraint<T>::DataVecCoord &d_x2,
                                const  typename BilateralInteractionConstraint<T>::DataVecDeriv &/*v1*/,
                                const  typename BilateralInteractionConstraint<T>::DataVecDeriv &/*v2*/)
    {
        const helper::vector<int> &m1Indices = self.m1.getValue();
        const helper::vector<int> &m2Indices = self.m2.getValue();

        unsigned min = std::min(m1Indices.size(), m2Indices.size());
        const  typename BilateralInteractionConstraint<T>::VecDeriv& restVector = self.restVector.getValue();
        self.dfree.resize(min);

        const  typename BilateralInteractionConstraint<T>::VecCoord &x1 = d_x1.getValue();
        const  typename BilateralInteractionConstraint<T>::VecCoord &x2 = d_x2.getValue();

        for (unsigned pid=0; pid<min; pid++)
        {
            //typename BilateralInteractionConstraint<T>::Coord dof1 = x1[m1Indices[pid]];
            //typename BilateralInteractionConstraint<T>::Coord dof2 = x2[m2Indices[pid]];
            typename BilateralInteractionConstraint<T>::Coord dof1;

             if (self.keepOrientDiff.getValue()) {
                 const typename BilateralInteractionConstraint<T>::Coord dof1c = x1[m1Indices[pid]];

                 typename BilateralInteractionConstraint<T>::Coord corr=self.initialDifference[pid];
                 defaulttype::Quat df = corr.getOrientation();
                 defaulttype::Quat o1 = dof1c.getOrientation();
                 defaulttype::Quat ro1 = o1 * df;

                 dof1.getCenter() = dof1c.getCenter() + corr.getCenter();
                 dof1.getOrientation() = ro1;
             } else
                 dof1 = x1[m1Indices[pid]];

            const typename BilateralInteractionConstraint<T>::Coord dof2 = x2[m2Indices[pid]];

            getVCenter(self.dfree[pid]) = dof2.getCenter() - dof1.getCenter();
            getVOrientation(self.dfree[pid]) =  dof1.rotate(self.q.angularDisplacement(dof2.getOrientation() ,
                                                                                  dof1.getOrientation())); // angularDisplacement compute the rotation vector btw the two quaternions
            if (pid < restVector.size())
                self.dfree[pid] -= restVector[pid];

            for (unsigned int i=0 ; i<self.dfree[pid].size() ; i++)
                v->set(self.cid[pid]+i, self.dfree[pid][i]);
        }
    }


    template <class T, typename MyClass = BilateralInteractionConstraint<T> >
    static void addContact(BilateralInteractionConstraint<T>& self, typename MyClass::Deriv /*norm*/,
                           typename MyClass::Coord P, typename MyClass::Coord Q,
                           typename MyClass::Real /*contactDistance*/, int m1, int m2,
                           typename MyClass::Coord /*Pfree*/, typename MyClass::Coord /*Qfree*/, long /*id*/, typename MyClass::PersistentID /*localid*/)
    {
        helper::WriteAccessor<Data<helper::vector<int> > > wm1 = self.m1;
        helper::WriteAccessor<Data<helper::vector<int> > > wm2 = self.m2;
        helper::WriteAccessor<Data<typename MyClass::VecDeriv > > wrest = self.restVector;
        wm1.push_back(m1);
        wm2.push_back(m2);

        typename MyClass::Deriv diff;
        getVCenter(diff) = Q.getCenter() - P.getCenter();
        getVOrientation(diff) =  P.rotate(self.q.angularDisplacement(Q.getOrientation() , P.getOrientation())) ; // angularDisplacement compute the rotation vector btw the two quaternions
        wrest.push_back(diff);
    }

};


#ifdef SOFA_WITH_DOUBLE
template<>
void BilateralInteractionConstraint<Rigid3dTypes>::init(){
    unspecializedInit() ;
}

template<>
void BilateralInteractionConstraint<Rigid3dTypes>::bwdInit() {
    BilateralInteractionConstraintSpecialization<RigidImpl>::bwdInit(*this);
}

template<>
void BilateralInteractionConstraint<Rigid3dTypes>::getConstraintResolution(const ConstraintParams* cParams,
                                                                           std::vector<ConstraintResolution*>& resTab,
                                                                           unsigned int& offset)
{
    BilateralInteractionConstraintSpecialization<RigidImpl>::getConstraintResolution(*this,
                                                                                     cParams, resTab, offset,
                                                                                     d_numericalTolerance.getValue()) ;
}

template <>
void BilateralInteractionConstraint<Rigid3dTypes>::buildConstraintMatrix(const ConstraintParams* cParams,
                                                                         DataMatrixDeriv &c1_d,
                                                                         DataMatrixDeriv &c2_d,
                                                                         unsigned int &constraintId,
                                                                         const DataVecCoord &x1, const DataVecCoord &x2)
{
    BilateralInteractionConstraintSpecialization<RigidImpl>::buildConstraintMatrix(*this,
                                                                                   cParams, c1_d, c2_d, constraintId,
                                                                                   x1, x2) ;
}


template <>
void BilateralInteractionConstraint<Rigid3dTypes>::getConstraintViolation(const ConstraintParams* cParams,
                                                                          BaseVector *v,
                                                                          const DataVecCoord &d_x1, const DataVecCoord &d_x2,
                                                                          const DataVecDeriv &v1, const DataVecDeriv &v2)
{
    BilateralInteractionConstraintSpecialization<RigidImpl>::getConstraintViolation(*this,
                                                                                    cParams, v, d_x1, d_x2,
                                                                                    v1, v2) ;
}


template <>
void BilateralInteractionConstraint<Rigid3dTypes>::getVelocityViolation(BaseVector * /*v*/,
                                                                        const DataVecCoord &/*x1*/,
                                                                        const DataVecCoord &/*x2*/,
                                                                        const DataVecDeriv &/*v1*/,
                                                                        const DataVecDeriv &/*v2*/)
{

}

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::addContact(Deriv norm,
                                                                           Coord P, Coord Q, Real contactDistance,
                                                                           int m1, int m2,
                                                                           Coord Pfree, Coord Qfree,
                                                                           long id, PersistentID localid)
{
    BilateralInteractionConstraintSpecialization<RigidImpl>::addContact(*this,
                                                                        norm, P, Q, contactDistance, m1, m2, Pfree, Qfree,
                                                                        id, localid) ;
}
#endif

#ifdef SOFA_WITH_FLOAT
template<>
void BilateralInteractionConstraint<Rigid3fTypes>::init()
{
    unspecializedInit() ;

    if(!d_numericalTolerance.isSet()){
        msg_warning(this) << "The numericalTolerance attribute is not set. The default value is now 0.0001.  "
                             "This value is different to the one used in SOFA 1.0 if you are experiencing problems "
                             "in your scene you can set the attribute to the old value of 0.01.  "
                             "To remove this warning message you need to explicitely specify the desired tolerance in"
                             "your scene.  ";
    }
}

template<>
void BilateralInteractionConstraint<Rigid3fTypes>::bwdInit() {
    BilateralInteractionConstraintSpecialization<RigidImpl>::bwdInit(*this);
}


template<>
void BilateralInteractionConstraint<Rigid3fTypes>::getConstraintResolution(const ConstraintParams* cParams,
                                                                           std::vector<ConstraintResolution*>& resTab,
                                                                           unsigned int& offset)
{

    BilateralInteractionConstraintSpecialization<RigidImpl>::getConstraintResolution(*this,
                                                                                     cParams, resTab, offset,
                                                                                     d_numericalTolerance.getValue());
}

template <>
void BilateralInteractionConstraint<Rigid3fTypes>::buildConstraintMatrix(const ConstraintParams* cParams,
                                                                         DataMatrixDeriv &c1_d,
                                                                         DataMatrixDeriv &c2_d,
                                                                         unsigned int &constraintId,
                                                                         const DataVecCoord &x1, const DataVecCoord &x2)
{
    BilateralInteractionConstraintSpecialization<RigidImpl>::buildConstraintMatrix(*this,
                                                                                   cParams, c1_d, c2_d, constraintId,
                                                                                   x1, x2) ;
}


template <>
void BilateralInteractionConstraint<Rigid3fTypes>::getConstraintViolation(const ConstraintParams* cParams,
                                                                          BaseVector *v,
                                                                          const DataVecCoord &d_x1,
                                                                          const DataVecCoord &d_x2,
                                                                          const DataVecDeriv &v1, const DataVecDeriv &v2)
{
    BilateralInteractionConstraintSpecialization<RigidImpl>::getConstraintViolation(*this,
                                                                                    cParams, v, d_x1, d_x2,
                                                                                    v1, v2) ;
}

template <>
void BilateralInteractionConstraint<Rigid3fTypes>::getVelocityViolation(BaseVector * /*v*/,
                                                                        const DataVecCoord &/*x1*/,
                                                                        const DataVecCoord &/*x2*/,
                                                                        const DataVecDeriv &/*v1*/,
                                                                        const DataVecDeriv &/*v2*/)
{

}

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::addContact(Deriv norm,
                                                                           Coord P, Coord Q,
                                                                           Real contactDistance,
                                                                           int m1, int m2, Coord Pfree,
                                                                           Coord Qfree,
                                                                           long id, PersistentID localid)
{
    BilateralInteractionConstraintSpecialization<RigidImpl>::addContact(*this,
                                                                        norm, P, Q, contactDistance, m1, m2, Pfree, Qfree,
                                                                        id, localid) ;
}

#endif


SOFA_DECL_CLASS(BilateralInteractionConstraint)

int BilateralInteractionConstraintClass = core::RegisterObject("TODO-BilateralInteractionConstraint")
        #ifdef SOFA_WITH_DOUBLE
        .add< BilateralInteractionConstraint<Vec3dTypes> >()
        .add< BilateralInteractionConstraint<Rigid3dTypes> >()
        #endif
        #ifdef SOFA_WITH_FLOAT
        .add< BilateralInteractionConstraint<Vec3fTypes> >()
        .add< BilateralInteractionConstraint<Rigid3fTypes> >()
        #endif
        ;

#ifdef SOFA_WITH_DOUBLE
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Vec3dTypes>;
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Rigid3dTypes>;
#endif
#ifdef SOFA_WITH_FLOAT
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Vec3fTypes>;
template class SOFA_CONSTRAINT_API BilateralInteractionConstraint<Rigid3fTypes>;
#endif

}

} // namespace constraintset

} // namespace component

} // namespace sofa

