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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_LAGRANGEMULTIPLIERINTERACTION_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_LAGRANGEMULTIPLIERINTERACTION_INL

#include <sofa/component/interactionforcefield/LagrangeMultiplierInteraction.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaBaseLinearSolver/FullVector.h>
#include <sofa/helper/system/config.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/core/VecId.h>


namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes1, class DataTypes2>
void LagrangeMultiplierInteraction<DataTypes1, DataTypes2>::init()
{
    core::behavior::BaseInteractionForceField::init();

    core::objectmodel::BaseContext* test = this->getContext();

    // ancien mécanisme: une seule contrainte
    test->get(constraint,f_constraint.getValue());


    // nouveau mécanisme: plusieurs contraintes
    simulation::FindByTypeVisitor< core::behavior::BaseConstraint > findConstraint(ExecParams::defaultInstance());
    findConstraint.execute(this->getContext());
    list_base_constraint = findConstraint.found;


    // debug //
    //sout<<"************** list of constraints : *************"<<sendl;
    long id[100];
    unsigned int offset;
    for (unsigned int i=0; i<list_base_constraint.size(); i++)
    {
        sout<<list_base_constraint[i]->getName()<<sendl;

        baseConstraint* bc =list_base_constraint[i];

        SimpleConstraint * sc = dynamic_cast<SimpleConstraint *>(bc);
        if (sc != NULL)
        {
            //debug
            //serr<< "simple constraint applied on object "<<sc->getMState()->getName() <<sendl;
            list_constraint.push_back(sc);
        }

        BaseInteractionConstraint* ic= dynamic_cast<BaseInteractionConstraint*> (bc);
        if (ic != NULL)
        {
            // debug
            //serr<< "interaction constraint applied on object "<<ic->getMechModel1()->getName() <<"  and on object "<< ic->getMechModel2()->getName()<<sendl;
            //serr<< "mstate1 : "<<this->mstate2->getName()<<sendl;

            core::objectmodel::BaseContext* context_model1 = ic->getMechModel1()->getContext();
            core::objectmodel::BaseContext* context_model2 = ic->getMechModel2()->getContext();
            if (this->mstate2==context_model1->getMechanicalState() || this->mstate2==context_model2->getMechanicalState() )
            {
                sout<<" - this constraint must be handled - "<<sendl;
                list_interaction_constraint.push_back(ic);

                offset=0;
                //	ic->getConstraintId(id, offset);
                sout<< "constraint offset"<<offset<<sendl;
            }
        }
    }

    for (unsigned int i=0; i<offset; i++)
    {
        sout<< "id : "<< id[i] <<sendl;
    }
}


template<class DataTypes1, class DataTypes2>
void LagrangeMultiplierInteraction<DataTypes1, DataTypes2>::addForce(
    const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv1& data_violation, DataVecDeriv2& /*data_f2*/,
    const DataVecCoord1& /*data_lambda*/ , const DataVecCoord2& /*data_p2*/,
    const DataVecDeriv1& , const DataVecDeriv2& /*data_v2*/)
{
    helper::WriteAccessor< DataVecDeriv1 > violation = data_violation;
    using linearsolver::FullVector;

    unsigned int count=0;

    for (unsigned int i=0; i<list_interaction_constraint.size(); i++)
    {
        list_interaction_constraint[i]->buildConstraintMatrix(count, core::VecCoordId::position());
        sout<< "constraint count"<<count<<sendl;
    }

    unsigned int count1=0;
    constraint->buildConstraintMatrix(count1, core::VecCoordId::position());


    /// @TODO clear the MechanicalState of the lagrange Multiplier during Begin visitor
    // clear the mechanical state of lagrange Multiplier
    // should be done elsewhere //
    helper::WriteAccessor<Data<VecCoord1> > lambda = *this->mstate1->write(sofa::core::VecCoordId::position());
    //VecCoord1& lambda= *this->mstate1->getX();
    unsigned int numLagMult = lambda.size();
    lambda.clear();
    lambda.resize(numLagMult);
    // clear the Velocity ? useful ?
    helper::WriteAccessor<Data<VecDeriv1> > Dlambda = *this->mstate1->write(sofa::core::VecDerivId::velocity());
    //VecDeriv1& Dlambda= *this->mstate1->getV();
    Dlambda.clear();
    Dlambda.resize(numLagMult);

    FullVector<double> _violation;

    /// @TODO automatically adapt the size of the LagMult state to the num of constraints: for now it is based the scene file entries
    _violation.resize(numLagMult);

//	constraint->getConstraintValue(&_violation, false);
    constraint->getConstraintViolation(&_violation, (VecId)core::VecCoordId::position());
    //sout<<"violation:" <<_violation[0] << " "<<_violation[1] << " "<<_violation[2] << " "<<sendl;

    for (unsigned int i=0; i<lambda.size(); i++)
    {
        violation[i].x() = (Real1)_violation[i];
    }
}


template<class DataTypes1, class DataTypes2>
void LagrangeMultiplierInteraction<DataTypes1, DataTypes2>::addDForce(
    const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv1& data_dViolation, DataVecDeriv2& data_df2,
    const DataVecDeriv1& data_dLambda, const DataVecDeriv2& data_dx2)
{

    helper::WriteAccessor< DataVecDeriv1 > dViolation = data_dViolation;
    helper::WriteAccessor< DataVecDeriv2 > df2        = data_df2;
    helper::ReadAccessor< DataVecDeriv1 >  dLambda    = data_dLambda;
    helper::ReadAccessor< DataVecDeriv2 >  dx2        = data_dx2;


    using sofa::simulation::Node;
    //sout<<"addDForce : dLambda "<< dLambda << " -  dx2:" << dx2 <<sendl;

    Node *context = dynamic_cast< Node* >(this->getContext()); // access to current node (which is supposed to be the root)
    sofa::simulation::MechanicalResetConstraintVisitor(mparams).execute(context);

    const MatrixDeriv2& c2 = *this->mstate2->getC();

    MatrixDeriv2RowConstIterator rowIt = c2.begin();
    MatrixDeriv2RowConstIterator rowItEnd = c2.end();

    unsigned int i = 0;

    while (rowIt != rowItEnd)
    {
        MatrixDeriv2ColConstIterator colIt = rowIt.begin();
        MatrixDeriv2ColConstIterator colItEnd = rowIt.end();

        while (colIt != colItEnd)
        {
            /// @TODO : use the constraint ID
            unsigned int index = colIt.index();
            Deriv2 value = colIt.val();
            dViolation[i].x() += (Real1)(value * dx2[index]);
            df2[index] += value * dLambda[i].x();

            ++colIt;
        }

        i++;
        ++rowIt;
    }

    //sout<<"addDForce : dViolation "<< dViolation << " -  df2:" << df2 <<sendl;
}

/*
template <class DataTypes1, class DataTypes2>
double LagrangeMultiplierInteraction<DataTypes1, DataTypes2>::getPotentialEnergy(const VecCoord1& x1 , const VecCoord2& x2 ) const
{
    serr<<"LagrangeMultiplierInteraction::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
*/

/*
template<class DataTypes1, class DataTypes2>
void LagrangeMultiplierInteraction<DataTypes1, DataTypes2>::draw(const core::visual::VisualParams* vparams)
{

}
*/

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_INTERACTIONFORCEFIELD_LAGRANGEMULTIPLIERINTERACTION_INL
