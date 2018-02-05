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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_DistanceLMContactConstraint_INL
#define SOFA_COMPONENT_CONSTRAINTSET_DistanceLMContactConstraint_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseLMConstraint.h>
#include <sofa/core/collision/Intersection.h>
#include <SofaConstraint/DistanceLMContactConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/accessor.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace constraintset
{

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
void DistanceLMContactConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* cParams, core::MultiMatrixDerivId cId, unsigned int &cIndex)
{
    using namespace core::objectmodel;
    Data<MatrixDeriv>* dC1 = cId[this->constrainedObject1].write();
    helper::WriteAccessor<Data<MatrixDeriv> > c1 = *dC1;

    Data<MatrixDeriv>* dC2 = cId[this->constrainedObject2].write();
    helper::WriteAccessor<Data<MatrixDeriv> > c2 = *dC2;

    helper::ReadAccessor<Data<VecCoord> > x1 = *cParams->readX(this->constrainedObject1);
    helper::ReadAccessor<Data<VecCoord> > x2 = *cParams->readX(this->constrainedObject2);

    const SeqEdges &edges =  pointPairs.getValue();

    //if (this->l0.size() != edges.size()) updateRestLength();

    scalarConstraintsIndices.clear();
    constraintGroupToContact.clear();
    edgeToContact.clear();

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        unsigned int idx1=edges[i][0];
        unsigned int idx2=edges[i][1];

        const Deriv normal = computeNormal(edges[i], x1.ref(), x2.ref());

        MatrixDerivRowIterator c1_normal = c1->writeLine(cIndex);
        c1_normal.addCol(idx1,normal);
        MatrixDerivRowIterator c2_normal = c2->writeLine(cIndex);
        c2_normal.addCol(idx2,-normal);
        scalarConstraintsIndices.push_back(cIndex++);

        Deriv tgt1, tgt2;
        computeTangentVectors(tgt1,tgt2,normal);

        MatrixDerivRowIterator c1_t1 = c1->writeLine(cIndex);
        c1_t1.addCol(idx1,tgt1);
        MatrixDerivRowIterator c2_t1 = c2->writeLine(cIndex);
        c2_t1.addCol(idx2,-tgt1);
        scalarConstraintsIndices.push_back(cIndex++);


        MatrixDerivRowIterator c1_t2 = c1->writeLine(cIndex);
        c1_t2.addCol(idx1,tgt2);
        MatrixDerivRowIterator c2_t2 = c2->writeLine(cIndex);
        c2_t2.addCol(idx2,-tgt2);
        scalarConstraintsIndices.push_back(cIndex++);

        this->constrainedObject1->forceMask.insertEntry(idx1);
        this->constrainedObject2->forceMask.insertEntry(idx2);

        edgeToContact[edges[i]] = Contact(normal,tgt1,tgt2);
    }

}
template<class DataTypes>
void DistanceLMContactConstraint<DataTypes>::writeConstraintEquations(unsigned int& lineNumber, core::MultiVecId id, ConstOrder Order)
{
    using namespace core;
    using namespace core::objectmodel;
    const SeqEdges &edges =  pointPairs.getValue();

    if (scalarConstraintsIndices.empty()) return;
    unsigned scalarConstraintIndex = 0;
    for (unsigned int i=0; i<edges.size(); ++i)
    {
        switch(Order)
        {
        case core::ConstraintParams::ACC :
        case core::ConstraintParams::VEL :
        {
            ConstVecId v1 = id.getId(this->simulatedObject1);
            ConstVecId v2 = id.getId(this->simulatedObject2);
            core::behavior::ConstraintGroup *constraintGroup = this->addGroupConstraint(Order);
            constraintGroupToContact[constraintGroup] = &edgeToContact[edges[i]];

            SReal correction = 0;
            correction+= this->simulatedObject1->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],v1);
            correction+= this->simulatedObject2->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],v2);
            constraintGroup->addConstraint( lineNumber, scalarConstraintsIndices[scalarConstraintIndex++], -correction);

            correction = 0;
            correction+= this->simulatedObject1->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],v1);
            correction+= this->simulatedObject2->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],v2);
            constraintGroup->addConstraint( lineNumber, scalarConstraintsIndices[scalarConstraintIndex++], -correction);

            correction = 0;
            correction+= this->simulatedObject1->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],v1);
            correction+= this->simulatedObject2->getConstraintJacobianTimesVecDeriv(scalarConstraintsIndices[scalarConstraintIndex],v2);
            constraintGroup->addConstraint( lineNumber, scalarConstraintsIndices[scalarConstraintIndex++], -correction);
            break;
        }
        case core::ConstraintParams::POS :
        case core::ConstraintParams::POS_AND_VEL :
        {
            SReal minDistance = 0;

            if (!intersection)
                this->getContext()->get(intersection);

            if (intersection)
                minDistance=intersection->getContactDistance();
            else
                serr << "No intersection component found!!" << sendl;

            const VecCoord &x1 = this->constrainedObject1->read(core::ConstVecCoordId(id.getId(this->constrainedObject1)))->getValue();
            const VecCoord &x2 = this->constrainedObject2->read(core::ConstVecCoordId(id.getId(this->constrainedObject2)))->getValue();

            SReal correction  = minDistance-lengthEdge(edges[i],x1,x2); //Distance min-current length

            if (correction>0)
            {
                core::behavior::ConstraintGroup *constraintGroup = this->addGroupConstraint(Order);
                constraintGroup->addConstraint( lineNumber, scalarConstraintsIndices[scalarConstraintIndex], correction);
            }

            scalarConstraintIndex += 3;
            break;
        }
        };

    }
}

template<class DataTypes>
void DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation(const SReal* W, const SReal* c, SReal* Lambda,
        core::behavior::ConstraintGroup * group)
{
    switch (group->getOrder())
    {
    case core::ConstraintParams::ACC :
    case core::ConstraintParams::VEL :
    {
        Contact &out=*(this->constraintGroupToContact[group]);
        ContactDescription &contact=this->getContactDescription(group);

        //                        //The force cannot be attractive!
        if (Lambda[0] <= 0)
        {
            contact.state=VANISHING;
            group->setActive(false);
            out.contactForce=Deriv();
            //                            msg_info()<<"DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation, deactivate attractive force"<<std::endl;
            return;
        }

        //Friction force

        const SReal normTangentForce=sqrt(pow(Lambda[1],2)+pow(Lambda[2],2));
        const SReal normNormalForce=fabs(Lambda[0]);

        const SReal& coeffFriction = contactFriction.getValue();

        //Test if we are outside the coulomb friction cone
        if ( (normTangentForce / normNormalForce) > coeffFriction)
        {
            contact.state=SLIDING;
            //Force applied
            out.contactForce = out.n*Lambda[0]+out.t1*Lambda[1]+ out.t2*Lambda[2];
            //Outside: we project the force to the cone

            //directionCone <--> n' : unitary vector along the cone
            const SReal factor=coeffFriction*normNormalForce/normTangentForce;

            Deriv directionCone=out.n*Lambda[0]+(out.t1*Lambda[1]+ out.t2*Lambda[2])*factor;
            directionCone.normalize();

            contact.coeff[0]=out.n  *directionCone;
            contact.coeff[1]=out.t1 *directionCone;
            contact.coeff[2]=out.t2 *directionCone;

            const SReal value=W[0]*contact.coeff[0]+
                    W[1]*contact.coeff[1] +
                    W[2]*contact.coeff[2];

            if (value == 0)
            {
                serr << "ERROR DIVISION BY ZERO AVOIDED: w=[" << W[0]  << "," << W[1] << "," << W[2]  << "] " << " DIRECTION CONE: " << directionCone << " BARY COEFF: " << contact.coeff[0] << ", " <<  contact.coeff[1] << ", " <<  contact.coeff[2] << std::endl;
                group->setActive(false);
                out.contactForce=Deriv();
                return;
            }
            const SReal slidingLambda=c[0]/value;
            out.contactForce = directionCone*slidingLambda;
            //Then project the force to the border of the cone
            Lambda[0]=out.contactForce*out.n;
            Lambda[1]=out.contactForce*out.t1;
            Lambda[2]=out.contactForce*out.t2;

            //                                msg_info()<<"DistanceLMContactConstraint<DataTypes>::LagrangeMultiplierEvaluation, , friction = "<<contactFriction.getValue()<<std::endl<<", cut excessive friction force, bounded Lambda = "<<std::endl<<Lambda<<std::endl;

        }
        else contact.state=STICKING;

        out.contactForce = out.n*Lambda[0]+out.t1*Lambda[1]+out.t2*Lambda[2];


        break;
    }
    case core::ConstraintParams::POS :
    case core::ConstraintParams::POS_AND_VEL :
    {
        //The force cannot be attractive!
        if (Lambda[0] < 0)
        {
            group->setActive(false);
            return;
        }
    }
    default: {}
    }

    return;
}


template<class DataTypes>
bool DistanceLMContactConstraint<DataTypes>::isCorrectionComputedWithSimulatedDOF(ConstOrder order) const
{
    switch(order)
    {
    case core::ConstraintParams::ACC :
    case core::ConstraintParams::VEL :
        return true;
    case core::ConstraintParams::POS :
    case core::ConstraintParams::POS_AND_VEL :
        return false;
    }
    return false;
}

template <class DataTypes>
void DistanceLMContactConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    //if (this->l0.size() != pointPairs.getValue().size()) updateRestLength();

    if (vparams->displayFlags().getShowBehaviorModels())
    {
        const VecCoord &x1= this->constrainedObject1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord &x2= this->constrainedObject2->read(core::ConstVecCoordId::position())->getValue();


        sofa::helper::vector< sofa::helper::vector< sofa::defaulttype::Vector3 > > points;
        points.resize(3);
        //Sliding: show direction of the new constraint
        sofa::helper::vector< sofa::defaulttype::Vector3 > slidingConstraints;

        const sofa::helper::vector< ConstraintGroup* > &groups = this->getConstraintsOrder(core::ConstraintParams::VEL);

        const SeqEdges &edges =  pointPairs.getValue();
        for (unsigned int i=0; i<groups.size(); ++i)
        {
            ContactDescription &contactDescription=this->getContactDescription(groups[i]);
            Contact &contactDirection=*(this->constraintGroupToContact[groups[i]]);

            points[contactDescription.state].push_back(x1[edges[i][0]]);
            points[contactDescription.state].push_back(x2[edges[i][1]]);

            if (contactDescription.state == SLIDING)
            {
                sofa::defaulttype::Vector3 direction=contactDirection.n *contactDescription.coeff[0] +
                        contactDirection.t1*contactDescription.coeff[1]+
                        contactDirection.t2*contactDescription.coeff[2];

                direction.normalize();
                SReal sizeV=this->lengthEdge(edges[i], x1,x2);

                slidingConstraints.push_back(x1[edges[i][0]]);
                slidingConstraints.push_back(x1[edges[i][0]]-direction*sizeV);
            }
        }

        vparams->drawTool()->drawLines(points[VANISHING], 1, colorsContactState[VANISHING]);
        vparams->drawTool()->drawLines(points[STICKING], 1, colorsContactState[STICKING]);
        vparams->drawTool()->drawLines(points[SLIDING], 1, colorsContactState[SLIDING]);

        vparams->drawTool()->drawLines(slidingConstraints, 1, colorsContactState.back());


    }
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif


