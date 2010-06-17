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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_DistanceLMContactConstraint_INL
#define SOFA_COMPONENT_CONSTRAINTSET_DistanceLMContactConstraint_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseLMConstraint.h>
#include <sofa/component/constraintset/DistanceLMContactConstraint.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/gl/template.h>
#include <iostream>





namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace core::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::behavior;


template <class DataTypes>
double DistanceLMContactConstraint<DataTypes>::lengthEdge(const Edge &e, const VecCoord &x1, const VecCoord &x2) const
{
    return (x2[e[1]] -  x1[e[0]]).norm();
}

template <class DataTypes>
void DistanceLMContactConstraint<DataTypes>::clear()
{
    pointPairs.beginEdit()->clear();
    pointPairs.endEdit();
}

template <class DataTypes>
void DistanceLMContactConstraint<DataTypes>::addContact(unsigned m1, unsigned m2)
{
    pointPairs.beginEdit()->push_back(Edge(m1,m2));
    pointPairs.endEdit();
}


template<class DataTypes>
typename DataTypes::Deriv DistanceLMContactConstraint<DataTypes>::computeNormal(const Edge &e, const VecCoord &x1, const VecCoord &x2) const
{
    //Deriv V12 = (x2[e[1]] - x1[e[0]]);
    Deriv V12 = (x1[e[0]] - x2[e[1]]);
    V12.normalize();
    return V12;
}

// Compute T1 and T2, normalized and orthogonal to N. N must be normalized already.
template<class DataTypes>
void DistanceLMContactConstraint<DataTypes>::computeTangentVectors( Deriv& T1, Deriv& T2, const Deriv& N )
{
    T1 = cross( N,Deriv(1,0,0) );
    if (dot(T1,T1) < 1.0e-2) T1=cross( N,Deriv(0,1,0) );
    T1.normalize();
    T2 = N.cross(T1);
}


template<class DataTypes>
void DistanceLMContactConstraint<DataTypes>::buildConstraintMatrix(unsigned int &constraintId, core::VecId position)
{
//                cerr<<"DistanceLMContactConstraint<DataTypes>::buildJacobian"<<endl;
    const VecCoord &x1=*(this->constrainedObject1->getVecCoord(position.index));
    const VecCoord &x2=*(this->constrainedObject2->getVecCoord(position.index));

    const SeqEdges &edges =  pointPairs.getValue();

    //if (this->l0.size() != edges.size()) updateRestLength();

    scalarConstraintsIndices.clear();
    constraintGroupToContact.clear();
    edgeToContact.clear();

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        unsigned int idx1=edges[i][0];
        unsigned int idx2=edges[i][1];

        const Deriv normal = computeNormal(edges[i], x1, x2);

        SparseVecDeriv V1; V1.add(idx1, normal);
        SparseVecDeriv V2; V2.add(idx2,-normal);
        registerEquationInJ1(constraintId,V1);
        registerEquationInJ2(constraintId,V2);
        scalarConstraintsIndices.push_back(constraintId++);

        Deriv tgt1, tgt2;
        computeTangentVectors(tgt1,tgt2,normal);
//                    cerr<<"DistanceLMContactConstraint<DataTypes>::buildJacobian, tgt1 = "<<tgt1<<", tgt2 = "<<tgt2<<endl;

        SparseVecDeriv T1V1; T1V1.add(idx1, tgt1);
        SparseVecDeriv T1V2; T1V2.add(idx2,-tgt1);
        registerEquationInJ1(constraintId,T1V1);
        registerEquationInJ2(constraintId,T1V2);
        scalarConstraintsIndices.push_back(constraintId++);

        SparseVecDeriv T2V1; T2V1.add(idx1, tgt2);
        SparseVecDeriv T2V2; T2V2.add(idx2,-tgt2);
        registerEquationInJ1(constraintId,T2V1);
        registerEquationInJ2(constraintId,T2V2);
        scalarConstraintsIndices.push_back(constraintId++);

        this->constrainedObject1->forceMask.insertEntry(idx1);
        this->constrainedObject2->forceMask.insertEntry(idx2);

        edgeToContact[edges[i]] = Contact(normal,tgt1,tgt2);
    }
}


template<class DataTypes>
void DistanceLMContactConstraint<DataTypes>::writeConstraintEquations(VecId id, ConstOrder Order)
{
//                cerr<<"DistanceLMContactConstraint<DataTypes>::writeConstraintEquations, scalarConstraintsIndices.size() = "<<scalarConstraintsIndices.size()<<endl;
    typedef core::behavior::BaseMechanicalState::VecId VecId;
    const SeqEdges &edges =  pointPairs.getValue();

    if (scalarConstraintsIndices.empty()) return;
    unsigned scalarConstraintIndex = 0;
    for (unsigned int i=0; i<edges.size(); ++i)
    {
        core::behavior::BaseLMConstraint::ConstraintGroup *constraintGroup = this->addGroupConstraint(Order);
        constraintGroupToContact[constraintGroup] = &edgeToContact[edges[i]];

        switch(Order)
        {
        case core::behavior::BaseConstraintSet::ACC :
        case core::behavior::BaseConstraintSet::VEL :
        {
            SReal correction = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],id);
            correction+= this->constrainedObject2->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],id);
            constraintGroup->addConstraint( scalarConstraintsIndices[scalarConstraintIndex++], -correction);
            correction = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],id);
            correction+= this->constrainedObject2->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],id);
            constraintGroup->addConstraint( scalarConstraintsIndices[scalarConstraintIndex++], -correction);
            correction = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],id);
            correction+= this->constrainedObject2->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],id);
            constraintGroup->addConstraint( scalarConstraintsIndices[scalarConstraintIndex++], -correction);
//                            cerr<<"DistanceLMContactConstraint<DataTypes>::writeConstraintEquations, constraint inserted "<<endl;
            break;
        }
        case core::behavior::BaseConstraintSet::POS :
        {
            SReal minDistance=0;
            if (!intersection) this->getContext()->get(intersection);
            if (intersection) minDistance=intersection->getContactDistance();
            else serr << "No intersection component found!!" << sendl;

            const VecCoord &x1=*(this->constrainedObject1->getVecCoord(id.index));
            const VecCoord &x2=*(this->constrainedObject2->getVecCoord(id.index));
            SReal correction  = minDistance-lengthEdge(edges[i],x1,x2); //Distance min-current length
            if (correction>0) constraintGroup->addConstraint( scalarConstraintsIndices[scalarConstraintIndex], correction);
            scalarConstraintIndex+=3;
            break;
        }
        };

    }
}

template<class DataTypes>
void DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation(const SReal* Wptr, SReal* cptr, SReal* LambdaInitptr,
        core::behavior::BaseLMConstraint::ConstraintGroup * group)
{
    const int numConstraintToProcess = group->getNumConstraint();
    Eigen::Map<VectorEigen> c(cptr, numConstraintToProcess);
    Eigen::Map<MatrixEigen> W(Wptr, numConstraintToProcess, numConstraintToProcess);
    Eigen::Map<VectorEigen> LambdaInit(LambdaInitptr, numConstraintToProcess);
    VectorEigen Lambda=sofa::component::linearsolver::LagrangeMultiplierComputation::ComputeLagrangeMultiplier(Wptr,cptr,LambdaInitptr,numConstraintToProcess);
//                std::cerr<<"DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation, c = "<< std::endl << c
//                        <<std::endl<<", LambdaInit = "<<LambdaInit<<std::endl<<std::endl<<", Lambda = "<<Lambda<<std::endl;

//                std::cerr<<"DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation, Lambda = "<<Lambda<<", friction = "<<contactFriction.getValue()<<std::endl;

    switch (group->getOrder())
    {
    case core::behavior::BaseConstraintSet::VEL :
    {
        Contact &out=*(this->constraintGroupToContact[group]);
//                        serr << "Lambda:" << Lambda.transpose() << sendl;
//                        //The force cannot be attractive!
        if (Lambda(0) <= 0)
        {
            group->setActive(false);
            out.contactForce=Deriv();
//                            std::cerr<<"DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation, deactivate attractive force"<<std::endl;
            return;
        }

        if (numConstraintToProcess == 3) //Friction force
        {
            const SReal normTangentForce=sqrt(pow(Lambda(1),2)+pow(Lambda(2),2));
            const SReal normNormalForce=fabs(Lambda(0));

            const SReal& coeffFriction = contactFriction.getValue();

            //Test if we are outside the coulomb friction cone
            if ( (normTangentForce / normNormalForce) > coeffFriction)
            {
                //Force applied
                out.contactForce = out.n*Lambda(0)+out.t1*Lambda(1)+ out.t2*Lambda(2);
                //Outside: we project the force to the cone

                //directionCone <--> n' : unitary vector along the cone
                const SReal factor=coeffFriction*normNormalForce/normTangentForce;

                Deriv directionCone=out.n*Lambda(0)+(out.t1*Lambda(1)+ out.t2*Lambda(2))*factor;
                directionCone.normalize();

                const SReal alpha=out.n  *directionCone;
                const SReal beta =out.t1 *directionCone;
                const SReal gamma=out.t2 *directionCone;

                const SReal value=W(0, 0)*alpha+
                        W(0, 1)*beta +
                        W(0, 2)*gamma;

                if (value == 0)
                {
                    serr << "ERROR DIVISION BY ZERO AVOIDED: w=[" << W(0, 0)  << "," << W(0, 1) << "," << W(0, 2)  << "] " << " DIRECTION CONE: " << directionCone << " BARY COEFF: " << alpha << ", " << beta << ", " << gamma << std::endl;
                    group->setActive(false);
                    out.contactForce=Deriv();
                    return;
                }
                const SReal slidingLambda=c(0)/value;
                out.contactForce = directionCone*slidingLambda;
                //Then project the force to the border of the cone
                Lambda(0)=out.contactForce*out.n;
                Lambda(1)=out.contactForce*out.t1;
                Lambda(2)=out.contactForce*out.t2;
//                                std::cerr<<"DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation, , friction = "<<contactFriction.getValue()<<std::endl<<", cut excessive friction force, bounded Lambda = "<<std::endl<<Lambda<<std::endl;

            }
            out.contactForce = out.n*Lambda(0)+out.t1*Lambda(1)+out.t2*Lambda(2);
        }
        else
            out.contactForce = out.n*Lambda(0);

        break;
    }
    case core::behavior::BaseConstraintSet::POS :
    {
        //The force cannot be attractive!
        if (Lambda(0) < 0)
        {
            group->setActive(false);
            return;
        }
    }
    default: {}
    }

    LambdaInit = Lambda;

    return;
}



#ifndef SOFA_FLOAT
template <>
void DistanceLMContactConstraint<defaulttype::Rigid3dTypes>::draw();
#endif
#ifndef SOFA_DOUBLE
template <>
void DistanceLMContactConstraint<defaulttype::Rigid3fTypes>::draw();
#endif

template <class DataTypes>
void DistanceLMContactConstraint<DataTypes>::draw()
{
    //if (this->l0.size() != pointPairs.getValue().size()) updateRestLength();

    if (this->getContext()->getShowBehaviorModels())
    {
        const VecCoord &x1=*(this->constrainedObject1->getX());
        const VecCoord &x2=*(this->constrainedObject2->getX());

        std::vector< Vector3 > points;
        const SeqEdges &edges =  pointPairs.getValue();
        for (unsigned int i=0; i<edges.size(); ++i)
        {
            points.push_back(x1[edges[i][0]]);
            points.push_back(x2[edges[i][1]]);
        }
        simulation::getSimulation()->DrawUtility.drawLines(points, 1, Vec<4,float>(0.0,1.0,0.0f,1.0f));
    }
}




} // namespace constraintset

} // namespace component

} // namespace sofa

#endif


