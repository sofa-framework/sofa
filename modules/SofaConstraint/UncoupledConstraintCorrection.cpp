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
#define SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_CPP

#include "UncoupledConstraintCorrection.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>

#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseMechanics/UniformMass.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;

template<>
SOFA_CONSTRAINT_API UncoupledConstraintCorrection< sofa::defaulttype::Rigid3Types >::UncoupledConstraintCorrection(sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types> *mm)
    : Inherit(mm)
    , compliance(initData(&compliance, "compliance", "Rigid compliance value: 1st value for translations, 6 others for upper-triangular part of symmetric 3x3 rotation compliance matrix"))
    , defaultCompliance(initData(&defaultCompliance, (Real)0.00001, "defaultCompliance", "Default compliance value for new dof or if all should have the same (in which case compliance vector should be empty)"))
    , f_verbose( initData(&f_verbose,false,"verbose","Dump the constraint matrix at each iteration") )
    , d_handleTopologyChange(initData(&d_handleTopologyChange, false, "handleTopologyChange", "Enable support of topological changes for compliance vector (should be disabled for rigids)"))
    , d_correctionVelocityFactor(initData(&d_correctionVelocityFactor, (Real)1.0, "correctionVelocityFactor", "Factor applied to the constraint forces when correcting the velocities"))
    , d_correctionPositionFactor(initData(&d_correctionPositionFactor, (Real)1.0, "correctionPositionFactor", "Factor applied to the constraint forces when correcting the positions"))
    , d_useOdeSolverIntegrationFactors(initData(&d_useOdeSolverIntegrationFactors, false, "useOdeSolverIntegrationFactors", "Use odeSolver integration factors instead of correctionVelocityFactor and correctionPositionFactor"))
    , m_pOdeSolver(NULL)
{
}

template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::init()
{
    Inherit::init();


    const VecReal& comp = compliance.getValue();

    VecReal usedComp;
    double  odeFactor = 1.0;

    this->getContext()->get(m_pOdeSolver);
    if (!m_pOdeSolver)
    {
        if (d_useOdeSolverIntegrationFactors.getValue() == true)
        {
            serr << "Can't find any odeSolver" << sendl;
            d_useOdeSolverIntegrationFactors.setValue(false);
        }
        d_useOdeSolverIntegrationFactors.setReadOnly(true);
    }
    else
    {
        if( !d_useOdeSolverIntegrationFactors.getValue() )
        {
            const double dt = this->getContext()->getDt();
            odeFactor = dt*dt; // W = h^2 * JMinvJt : only correct when solving in constraint equation in position. Must be deprecated.
        }
    }


    if (comp.size() != 7)
    {
        using sofa::component::mass::UniformMass;
        using sofa::defaulttype::Rigid3Types;
        using sofa::defaulttype::Rigid3Mass;
        using sofa::simulation::Node;

        Node *node = dynamic_cast< Node * >(getContext());
        Rigid3Mass massValue;

        //Should use the BaseMatrix API to get the Mass
        //void getElementMass(unsigned int index, defaulttype::BaseMatrix *m)
        if (node != NULL)
        {
            core::behavior::BaseMass *m = node->mass;
            UniformMass< Rigid3Types, Rigid3Mass > *um = dynamic_cast< UniformMass< Rigid3Types, Rigid3Mass >* > (m);

            if (um)
                massValue = um->getMass();
            else
                serr << "WARNING : no mass found" << sendl;
        }
        else
        {
            serr << "\n WARNING : node is not found => massValue could be incorrect in addComplianceInConstraintSpace function" << sendl;
        }
        

        if (defaultCompliance.isSet())
        {
            usedComp.push_back(defaultCompliance.getValue());
        }
        else
        {
            usedComp.push_back(odeFactor / massValue.mass);
        }

        usedComp.push_back( odeFactor * massValue.invInertiaMassMatrix[0][0]);
        usedComp.push_back( odeFactor * massValue.invInertiaMassMatrix[0][1]);
        usedComp.push_back( odeFactor * massValue.invInertiaMassMatrix[0][2]);
        usedComp.push_back( odeFactor * massValue.invInertiaMassMatrix[1][1]);
        usedComp.push_back( odeFactor * massValue.invInertiaMassMatrix[1][2]);
        usedComp.push_back( odeFactor * massValue.invInertiaMassMatrix[2][2]);
        compliance.setValue(usedComp);
    }
    else
    {
        sout << "COMPLIANCE VALUE FOUND" << sendl;
    }
    

}


template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::getComplianceMatrix(defaulttype::BaseMatrix *m) const
{
    const VecReal& comp = compliance.getValue();
    const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
    const unsigned int numDofs = comp.size() / 7;

    m->resize(dimension * numDofs, dimension * numDofs);

    /// @todo Optimization
    for (unsigned int d = 0; d < numDofs; ++d)
    {
        const unsigned int d6 = 6 * d;
        const unsigned int d7 = 7 * d;
        const SReal invM = comp[d7];

        m->set(d6, d6, invM);
        m->set(d6 + 1, d6 + 1, invM);
        m->set(d6 + 2, d6 + 2, invM);

        m->set(d6 + 3, d6 + 3, comp[d7 + 1]);

        m->set(d6 + 3, d6 + 4, comp[d7 + 2]);
        m->set(d6 + 4, d6 + 3, comp[d7 + 2]);

        m->set(d6 + 3, d6 + 5, comp[d7 + 3]);
        m->set(d6 + 5, d6 + 3, comp[d7 + 3]);

        m->set(d6 + 4, d6 + 4, comp[d7 + 4]);

        m->set(d6 + 4, d6 + 5, comp[d7 + 5]);
        m->set(d6 + 5, d6 + 4, comp[d7 + 5]);

        m->set(d6 + 5, d6 + 5, comp[d7 + 6]);
    }
}


SOFA_DECL_CLASS(UncoupledConstraintCorrection)

int UncoupledConstraintCorrectionClass = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
#ifndef SOFA_FLOAT
        .add< UncoupledConstraintCorrection< Vec1dTypes > >()
        .add< UncoupledConstraintCorrection< Vec2dTypes > >()
        .add< UncoupledConstraintCorrection< Vec3dTypes > >()
        .add< UncoupledConstraintCorrection< Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< UncoupledConstraintCorrection< Vec1fTypes > >()
        .add< UncoupledConstraintCorrection< Vec2fTypes > >()
        .add< UncoupledConstraintCorrection< Vec3fTypes > >()
        .add< UncoupledConstraintCorrection< Rigid3fTypes > >()
        //TODO(dmarchal) There is no Rigid3fTypes template specizaliation while there is one for Rigid3d...
        //this look sucipicious.

#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec1dTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec2dTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec3dTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec1fTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec2fTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec3fTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Rigid3fTypes >;
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa
