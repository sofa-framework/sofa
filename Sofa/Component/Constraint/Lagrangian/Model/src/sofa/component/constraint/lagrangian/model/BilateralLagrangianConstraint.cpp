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
    using SubsetIndices = typename BilateralLagrangianConstraint<T>::SubsetIndices;


    template<class T>
    static void bwdInit(BilateralLagrangianConstraint<T>& self) {
        if (!self.d_keepOrientDiff.getValue())
            return;

        auto wrest = sofa::helper::getWriteAccessor(self.d_restVector);

        if (!wrest.empty()) {
            msg_warning(&self) << "keepOrientationDifference is activated, rest_vector will be ignored! " ;
            wrest.resize(0);
        }

        const SubsetIndices<T>& m1Indices = self.d_m1.getValue();
        const SubsetIndices<T>& m2Indices = self.d_m2.getValue();

        const unsigned minp = std::min(m1Indices.size(),m2Indices.size());

        const DataVecCoord_t<T> &d_x1 = *self.mstate1->read(core::vec_id::read_access::position);
        const DataVecCoord_t<T> &d_x2 = *self.mstate2->read(core::vec_id::read_access::position);

        const VecCoord_t<T> &x1 = d_x1.getValue();
        const VecCoord_t<T> &x2 = d_x2.getValue();

        for (unsigned pid=0; pid<minp; pid++)
        {
            const Coord_t<T> P = x1[m1Indices[pid]];
            const Coord_t<T> Q = x2[m2Indices[pid]];

            type::Quat<SReal> qP, qQ, dQP;
            qP = P.getOrientation();
            qQ = Q.getOrientation();
            qP.normalize();
            qQ.normalize();
            dQP = qP.quatDiff(qQ, qP);
            dQP.normalize();

            Coord_t<T> df;
            df.getCenter() = Q.getCenter() - P.getCenter();
            df.getOrientation() = dQP;
            self.initialDifference.push_back(df);

        }
    }


    template<class T>
    static void getConstraintResolution(BilateralLagrangianConstraint<T>& self,
                                        const ConstraintParams* cParams,
                                        std::vector<ConstraintResolution*>& resTab,
                                        unsigned int& offset)
    {
        SOFA_UNUSED(cParams);
        const unsigned minp = std::min(self.d_m1.getValue().size(),
                                       self.d_m2.getValue().size());
        for (unsigned pid = 0; pid < minp; pid++)
        {
            resTab[offset] = new BilateralConstraintResolution3Dof();
            offset += 3;
            BilateralConstraintResolution3Dof* temp = new BilateralConstraintResolution3Dof();
            resTab[offset] = temp;
            offset += 3;
        }
    }


    template <class T>
    static void buildConstraintMatrix(BilateralLagrangianConstraint<T>& self,
                                      const ConstraintParams* cParams,
                                      DataMatrixDeriv_t<T> &c1_d,
                                      DataMatrixDeriv_t<T> &c2_d,
                                      unsigned int &constraintId,
                                      const DataVecCoord_t<T> &/*x1*/,
                                      const DataVecCoord_t<T> &/*x2*/)
    {
        SOFA_UNUSED(cParams) ;
        const SubsetIndices<T>& m1Indices = self.d_m1.getValue();
        const SubsetIndices<T>& m2Indices = self.d_m2.getValue();

        unsigned minp = std::min(m1Indices.size(),m2Indices.size());
        self.cid.resize(minp);

        auto c1 = sofa::helper::getWriteAccessor(c1_d);
        auto c2 = sofa::helper::getWriteAccessor(c2_d);

        static constexpr Vec<3, Real_t<T>> cx(1,0,0), cy(0,1,0), cz(0,0,1);
        static constexpr Vec<3, Real_t<T>> vZero(0,0,0);

        for (unsigned pid = 0; pid < minp; ++pid)
        {
            int tm1 = m1Indices[pid];
            int tm2 = m2Indices[pid];

            self.cid[pid] = constraintId;
            constraintId += 6;

            //Apply constraint for position
            auto c1_it = c1->writeLine(self.cid[pid]);
            c1_it.addCol(tm1, Deriv_t<T>(-cx, vZero));

            auto c2_it = c2->writeLine(self.cid[pid]);
            c2_it.addCol(tm2, Deriv_t<T>(cx, vZero));

            c1_it = c1->writeLine(self.cid[pid] + 1);
            c1_it.setCol(tm1, Deriv_t<T>(-cy, vZero));

            c2_it = c2->writeLine(self.cid[pid] + 1);
            c2_it.setCol(tm2, Deriv_t<T>(cy, vZero));

            c1_it = c1->writeLine(self.cid[pid] + 2);
            c1_it.setCol(tm1, Deriv_t<T>(-cz, vZero));

            c2_it = c2->writeLine(self.cid[pid] + 2);
            c2_it.setCol(tm2, Deriv_t<T>(cz, vZero));

            //Apply constraint for orientation
            c1_it = c1->writeLine(self.cid[pid] + 3);
            c1_it.setCol(tm1, Deriv_t<T>(vZero, -cx));

            c2_it = c2->writeLine(self.cid[pid] + 3);
            c2_it.setCol(tm2, Deriv_t<T>(vZero, cx));

            c1_it = c1->writeLine(self.cid[pid] + 4);
            c1_it.setCol(tm1, Deriv_t<T>(vZero, -cy));

            c2_it = c2->writeLine(self.cid[pid] + 4);
            c2_it.setCol(tm2, Deriv_t<T>(vZero, cy));

            c1_it = c1->writeLine(self.cid[pid] + 5);
            c1_it.setCol(tm1, Deriv_t<T>(vZero, -cz));

            c2_it = c2->writeLine(self.cid[pid] + 5);
            c2_it.setCol(tm2, Deriv_t<T>(vZero, cz));
        }
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
        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m1Indices = self.d_m1.getValue();
        const typename BilateralLagrangianConstraint<T>::SubsetIndices& m2Indices = self.d_m2.getValue();

        unsigned min = std::min(m1Indices.size(), m2Indices.size());
        const VecDeriv_t<T>& restVector = self.d_restVector.getValue();
        self.m_violation.resize(min);

        const VecCoord_t<T> &x1 = d_x1.getValue();
        const VecCoord_t<T> &x2 = d_x2.getValue();

        for (unsigned pid = 0; pid < min; pid++)
        {
            Coord_t<T> dof1;

            if (self.d_keepOrientDiff.getValue())
            {
                const Coord_t<T> dof1c = x1[m1Indices[pid]];

                Coord_t<T> corr = self.initialDifference[pid];
                type::Quat<Real_t<T>> df = corr.getOrientation();
                type::Quat<Real_t<T>> o1 = dof1c.getOrientation();
                type::Quat<Real_t<T>> ro1 = o1 * df;

                dof1.getCenter() = dof1c.getCenter() + corr.getCenter();
                dof1.getOrientation() = ro1;
            }
            else
            {
                dof1 = x1[m1Indices[pid]];
            }

            const Coord_t<T> dof2 = x2[m2Indices[pid]];

            getVCenter(self.m_violation[pid]) = dof2.getCenter() - dof1.getCenter();
            getVOrientation(self.m_violation[pid]) =  dof1.rotate(self.q.angularDisplacement(dof2.getOrientation() ,
                                                                                  dof1.getOrientation())); // angularDisplacement compute the rotation vector btw the two quaternions
            if (pid < restVector.size())
                self.m_violation[pid] -= restVector[pid];

            for (unsigned int i = 0; i < self.m_violation[pid].size(); i++)
            {
                v->set(self.cid[pid]+i, self.m_violation[pid][i]);
            }
        }
    }


    template <class T, typename MyClass = BilateralLagrangianConstraint<T> >
    static void addContact(BilateralLagrangianConstraint<T>& self, typename MyClass::Deriv /*norm*/,
                           typename MyClass::Coord P, typename MyClass::Coord Q,
                           typename MyClass::Real /*contactDistance*/, int m1, int m2,
                           typename MyClass::Coord /*Pfree*/, typename MyClass::Coord /*Qfree*/, long /*id*/, typename MyClass::PersistentID /*localid*/)
    {
        sofa::helper::WriteAccessor<Data<SubsetIndices<T>>> wm1 = self.d_m1;
        sofa::helper::WriteAccessor<Data<SubsetIndices<T>>> wm2 = self.d_m2;
        auto wrest = sofa::helper::WriteAccessor(self.d_restVector);
        wm1.push_back(m1);
        wm2.push_back(m2);

        typename MyClass::Deriv diff;
        getVCenter(diff) = Q.getCenter() - P.getCenter();
        getVOrientation(diff) =  P.rotate(self.q.angularDisplacement(Q.getOrientation() , P.getOrientation())) ; // angularDisplacement compute the rotation vector btw the two quaternions
        wrest.push_back(diff);
    }

};

using RigidBilateralLagrangianConstraint = BilateralLagrangianConstraintSpecialization<RigidImpl>;


template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::init()
{
    unspecializedInit() ;
}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::bwdInit()
{
    RigidBilateralLagrangianConstraint::bwdInit(*this);
}

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::getConstraintResolution(
    const ConstraintParams* cParams,
    std::vector<ConstraintResolution*>& resTab,
    unsigned int& offset)
{
    RigidBilateralLagrangianConstraint::getConstraintResolution(*this,
        cParams, resTab, offset);
}

template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::buildConstraintMatrix(const ConstraintParams* cParams,
                                                                         DataMatrixDeriv &c1_d,
                                                                         DataMatrixDeriv &c2_d,
                                                                         unsigned int &constraintId,
                                                                         const DataVecCoord &x1, const DataVecCoord &x2)
{
    RigidBilateralLagrangianConstraint::buildConstraintMatrix(*this,
        cParams, c1_d, c2_d, constraintId,
        x1, x2);
}


template <> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::getConstraintViolation(const ConstraintParams* cParams,
                                                                          BaseVector *v,
                                                                          const DataVecCoord &d_x1, const DataVecCoord &d_x2,
                                                                          const DataVecDeriv &v1, const DataVecDeriv &v2)
{
    RigidBilateralLagrangianConstraint::getConstraintViolation(*this,
        cParams, v, d_x1, d_x2,
        v1, v2);
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
    RigidBilateralLagrangianConstraint::addContact(*this,
                                                   norm, P, Q, contactDistance,
                                                   m1, m2, Pfree, Qfree,
                                                   id, localid);
}

void registerBilateralLagrangianConstraint(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("BilateralLagrangianConstraint defining an holonomic equality constraint (attachment).")
        .add< BilateralLagrangianConstraint<Vec3Types> >()
        .add< BilateralLagrangianConstraint<Rigid3Types> >());
}

template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralLagrangianConstraint<Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralLagrangianConstraint<Rigid3Types>;

} //namespace sofa::component::constraint::lagrangian::model
