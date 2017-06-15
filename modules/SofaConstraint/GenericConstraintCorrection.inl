/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_CORE_COLLISION_GENERICCONTACTCORRECTION_INL
#define SOFA_CORE_COLLISION_GENERICCONTACTCORRECTION_INL

#include "GenericConstraintCorrection.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseLinearSolver/GraphScatteredTypes.h>

#include <sstream>
#include <list>

namespace sofa {

namespace component {

namespace constraintset {

GenericConstraintCorrection::GenericConstraintCorrection()
: solverName( initData(&solverName, "solverName", "name of the constraint solver") )
{
    odesolver = NULL;
}

GenericConstraintCorrection::~GenericConstraintCorrection() {}

void GenericConstraintCorrection::bwdInit() {
    sofa::core::objectmodel::BaseContext* c = this->getContext();

    c->get(odesolver, core::objectmodel::BaseContext::SearchRoot);

    linearsolvers.clear();

    const helper::vector<std::string>& solverNames = solverName.getValue();
    for (unsigned int i=0; i<solverNames.size(); ++i) {
        sofa::core::behavior::LinearSolver* s = NULL;
        c->get(s, solverNames[i]);

        if (s) {
            if (s->getTemplateName() == "GraphScattered") {
                serr << "ERROR GenericConstraintCorrection cannot use the solver " << solverNames[i] << " because it is templated on GraphScatteredType" << sendl;
            } else {
                linearsolvers.push_back(s);
            }
        } else serr << "Solver \"" << solverNames[i] << "\" not found." << sendl;
    }

    if (odesolver == NULL) {
        serr << "GenericConstraintCorrection: ERROR no OdeSolver found."<<sendl;
        return;
    }

    if (linearsolvers.size()==0) {
        serr << "GenericConstraintCorrection: ERROR no LinearSolver found."<<sendl;
        return;
    }

    sout << "Found " << linearsolvers.size() << "linearsolvers" << sendl;
    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        sout << linearsolvers[i]->getName() << sendl;
    }
}

void GenericConstraintCorrection::cleanup()
{
    while(!constraintsolvers.empty())
    {
        constraintsolvers.back()->removeConstraintCorrection(this);
        constraintsolvers.pop_back();
    }
    sofa::core::behavior::BaseConstraintCorrection::cleanup();
}

void GenericConstraintCorrection::addConstraintSolver(core::behavior::ConstraintSolver *s)
{
    constraintsolvers.push_back(s);
}

void GenericConstraintCorrection::removeConstraintSolver(core::behavior::ConstraintSolver *s)
{
    constraintsolvers.remove(s);
}


void GenericConstraintCorrection::rebuildSystem(double massFactor, double forceFactor) {
    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->rebuildSystem(massFactor, forceFactor);
    }
}

void GenericConstraintCorrection::addComplianceInConstraintSpace(const sofa::core::ConstraintParams *cparams, sofa::defaulttype::BaseMatrix* W) {
    if (!odesolver) return;

    // use the OdeSolver to get the position integration factor
    double factor = 1.0;

    switch (cparams->constOrder())
    {
        case core::ConstraintParams::POS_AND_VEL :
        case core::ConstraintParams::POS :
            factor = odesolver->getPositionIntegrationFactor();
            break;

        case core::ConstraintParams::ACC :
        case core::ConstraintParams::VEL :
            factor = odesolver->getVelocityIntegrationFactor();
            break;

        default :
            break;
    }

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->buildComplianceMatrix(W, factor);
    }
}

void GenericConstraintCorrection::computeAndApplyMotionCorrection(const core::ConstraintParams * /*cparams*/, core::MultiVecCoordId /*xId*/, core::MultiVecDerivId /*vId*/, core::MultiVecDerivId /*fId*/, const defaulttype::BaseVector *lambda) {
    if (!odesolver) return;

    const double positionFactor = odesolver->getPositionIntegrationFactor();
    const double velocityFactor = odesolver->getVelocityIntegrationFactor();

    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->applyContactForce(lambda,positionFactor,velocityFactor);
    }
}


void GenericConstraintCorrection::computeResidual(const sofa::core::ExecParams* params, sofa::defaulttype::BaseVector *lambda) {
    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->computeResidual(params,lambda);
    }
}

void GenericConstraintCorrection::computeAndApplyPositionCorrection(const sofa::core::ConstraintParams * /*cparams*/, sofa::core::MultiVecCoordId /*xId*/, sofa::core::MultiVecDerivId /*fId*/, const sofa::defaulttype::BaseVector *lambda) {
    if (!odesolver) return;

    const double positionFactor = odesolver->getPositionIntegrationFactor();

    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->applyContactForce(lambda,positionFactor,0.0);
    }
}

void GenericConstraintCorrection::computeAndApplyVelocityCorrection(const sofa::core::ConstraintParams * /*cparams*/, sofa::core::MultiVecDerivId /*vId*/, sofa::core::MultiVecDerivId /*fId*/, const sofa::defaulttype::BaseVector *lambda) {
    if (!odesolver) return;

    const double velocityFactor = odesolver->getVelocityIntegrationFactor();

    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->applyContactForce(lambda,0.0,velocityFactor);
    }
}

void GenericConstraintCorrection::applyContactForce(const defaulttype::BaseVector *f) {
    if (!odesolver) return;

    const double positionFactor = odesolver->getPositionIntegrationFactor();
    const double velocityFactor = odesolver->getVelocityIntegrationFactor();

    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->applyContactForce(f,positionFactor,velocityFactor);
    }
}

void GenericConstraintCorrection::getComplianceMatrix(defaulttype::BaseMatrix* Minv) const {
    if (!odesolver) return;

    // use the OdeSolver to get the position integration factor
    double factor = odesolver->getPositionIntegrationFactor();

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    for (unsigned i = 0; i < linearsolvers.size(); i++) {
        linearsolvers[i]->buildComplianceMatrix(Minv, factor);
    }
}

void GenericConstraintCorrection::applyPredictiveConstraintForce(const core::ConstraintParams * /*cparams*/, core::MultiVecDerivId /*f*/, const defaulttype::BaseVector * /*lambda*/) {
//    printf("GenericConstraintCorrection::applyPredictiveConstraintForce not implemented\n");
//    if (mstate)
//    {
//        Data< VecDeriv > *f_d = f[mstate].write();

//        if (f_d)
//        {
//            applyPredictiveConstraintForce(cparams, *f_d, lambda);
//        }
//    }
}

void GenericConstraintCorrection::resetContactForce() {
//    printf("GenericConstraintCorrection::resetContactForce not implemented\n");
//    Data<VecDeriv>& forceData = *this->mstate->write(core::VecDerivId::force());
//    VecDeriv& force = *forceData.beginEdit();
//    for( unsigned i=0; i<force.size(); ++i )
//        force[i] = Deriv();
//    forceData.endEdit();
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
