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
#include <sofa/component/mastersolver/MasterConstraintSolver.h>
#include <sofa/component/mastersolver/MasterContactSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>

#include <sofa/helper/LCPcalc.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/system/thread/CTime.h>

#include <math.h>
#include <iostream>
#include <map>

namespace sofa
{

namespace component
{

namespace mastersolver
{

using namespace sofa::component::odesolver;
using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::componentmodel::behavior;



MasterConstraintSolver::MasterConstraintSolver()
    :displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of MasterConstraintSolver.")),
     _tol( initData(&_tol, 0.00001, "tolerance", "Tolerance of the Gauss-Seidel")),
     _maxIt( initData(&_maxIt, 1000, "maxIterations", "Maximum number of iterations of the Gauss-Seidel")),
     doCollisionsFirst(initData(&doCollisionsFirst, false, "doCollisionsFirst","Compute the collisions first (to support penality-based contacts)"))
{
}

MasterConstraintSolver::~MasterConstraintSolver()
{
}

void MasterConstraintSolver::init()
{
    getContext()->get<core::componentmodel::behavior::BaseConstraintCorrection> ( &constraintCorrections, core::objectmodel::BaseContext::SearchDown );
}

void MasterConstraintSolver::step ( double dt )
{
    CTime *timer;
    double time = 0.0, totaltime = 0.0;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec() * 1000;
    if ( displayTime.getValue() )
    {
        timer = new CTime();
        time = (double) timer->getTime();
        totaltime = time;
        sout<<sendl;
    }

    bool debug =this->f_printLog.getValue();
    if (debug)
        serr<<"MasterConstraintSolver::step is called"<<sendl;
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node


    if (doCollisionsFirst.getValue())
    {
        if (debug)
            serr<<"computeCollision is called"<<sendl;

        ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
        computeCollision();
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if ( displayTime.getValue() )
        {
            sout<<" computeCollision " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
            time = (double) timer->getTime();
        }
    }

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    simulation::BehaviorUpdatePositionVisitor(dt).execute(context);
    if (debug)
        serr<<"Free Motion is called"<<sendl;

    ///////////////////////////////////////////// FREE MOTION /////////////////////////////////////////////////////////////
    simulation::MechanicalBeginIntegrationVisitor(dt).execute(context);
    simulation::SolveVisitor(dt, true).execute(context);
    simulation::MechanicalPropagateFreePositionVisitor().execute(context);

    core::componentmodel::behavior::BaseMechanicalState::VecId dx_id = core::componentmodel::behavior::BaseMechanicalState::VecId::dx();
    simulation::MechanicalVOpVisitor(dx_id).execute(context);
    simulation::MechanicalPropagateDxVisitor(dx_id).execute(context);
    simulation::MechanicalVOpVisitor(dx_id).execute(context);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout << ">>>>> Begin display MasterContactSolver time" << sendl;
        sout<<" Free Motion                           " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (double) timer->getTime();
    }

    if (!doCollisionsFirst.getValue())
    {
        if (debug)
            serr<<"computeCollision is called"<<sendl;

        ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
        computeCollision();
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if ( displayTime.getValue() )
        {
            sout<<" ComputeCollision                      " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
            time = (double) timer->getTime();
        }
    }

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }


    //////////////////////////////////////CONSTRAINTS RESOLUTION//////////////////////////////////////////////////////////////////////
    if (debug)
        serr<<"constraints Matrix construction is called"<<sendl;

    unsigned int numConstraints = 0;

    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor().execute(context);
    // calling applyConstraint
//	simulation::MechanicalAccumulateConstraint(numConstraints).execute(context);
    MechanicalSetConstraint(numConstraints).execute(context);

    // calling accumulateConstraint
    MechanicalAccumulateConstraint2().execute(context);

    if (debug)
        serr<<"   1. resize constraints : numConstraints="<< numConstraints<<sendl;

    _dFree.resize(numConstraints);
    _d.resize(numConstraints);
    _W.resize(numConstraints,numConstraints);
    _constraintsType.resize(numConstraints);
    _force.resize(numConstraints);
    _constraintsResolutions.resize(numConstraints); // _constraintsResolutions.clear();

    if (debug)
        serr<<"   2. compute violation"<<sendl;
    // calling getConstraintValue
    MechanicalGetConstraintValueVisitor(&_dFree).execute(context);

    if (debug)
        serr<<"   3. get resolution method for each constraint"<<sendl;
    // calling getConstraintResolution
    MechanicalGetConstraintResolutionVisitor(_constraintsResolutions).execute(context);

    if (debug)
        serr<<"   4. get Compliance "<<sendl;

    for (unsigned int i=0; i<constraintCorrections.size(); i++ )
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getCompliance(&_W); // getDelassusOperator(_W) = H*C*Ht
    }

    if ( displayTime.getValue() )
    {
        sout<<" Build problem in the constraint space " << ( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer->getTime();
    }

    if (debug)
        serr<<"Gauss-Seidel solver is called"<<sendl;
    gaussSeidelConstraint(numConstraints, _dFree.ptr(), _W.lptr(), _force.ptr(), _d.ptr(), _constraintsResolutions);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ( displayTime.getValue() )
    {
        sout<<" Solve with GaussSeidel                " <<( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer->getTime();
    }

//	helper::afficheLCP(_dFree.ptr(), _W.lptr(), _force.ptr(),  numConstraints);

    if (debug)
        sout<<"constraintCorrections motion is called"<<sendl;

    ///////////////////////////////////////CORRECTIVE MOTION //////////////////////////////////////////////////////////////////////////
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(&_force);
    }

    simulation::MechanicalPropagateAndAddDxVisitor().execute(context);
    simulation::MechanicalPropagatePositionAndVelocityVisitor().execute(context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    if ( displayTime.getValue() )
    {
        sout<<" ContactCorrections                    " <<( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        sout<<"  = Total                              " <<( (double) timer->getTime() - totaltime)*timeScale <<" ms" <<sendl;
        sout << "<<<<< End display MasterContactSolver time." << sendl;
    }

    simulation::MechanicalEndIntegrationVisitor endVisitor(dt);
    context->execute(&endVisitor);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}

void MasterConstraintSolver::gaussSeidelConstraint(int dim, double* dfree, double** w, double* force,
        double* d, std::vector<ConstraintResolution*>& res)
{
//	sout<<"------------------------------------ new iteration ---------------------------------"<<sendl;
    int i, j, k, l, nb;

    double errF[6];
    double error=0.0;

    double tolerance = _tol.getValue();
    int numItMax = _maxIt.getValue();
    bool convergence = false;

    for(i=0; i<dim; )
    {
        res[i]->init(i, w, force);
        i += res[i]->nbLines;
    }

    for(i=0; i<numItMax; i++)
    {
        error=0.0;
        for(j=0; j<dim;)
        {
            nb = res[j]->nbLines;

            for(l=0; l<nb; l++)
            {
                errF[l] = force[j+l];
                d[j+l] = dfree[j+l];
            }

            for(k=0; k<dim; k++)
                for(l=0; l<nb; l++)
                    d[j+l] += w[j+l][k] * force[k];

            res[j]->resolution(j, w, d, force);

            if(nb > 1)
            {
                double terr = 0.0, terr2;
                for(l=0; l<nb; l++)
                {
                    terr2 = w[j+l][j+l] * (force[j+l] - errF[l]);
                    terr += terr2 * terr2;
                }
                error += sqrt(terr);
            }
            else
                error += fabs(w[j][j] * (force[j] - errF[0]));

            j += nb;
        }

        if(error < tolerance && i>0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    if(!convergence)
        serr << "------  No convergence in gaussSeidelConstraint : error = " << error <<" ------" <<sendl;
    else if ( displayTime.getValue() )
        sout<<" Convergence after " << i+1 << " iterations " << sendl;

    for(i=0; i<dim; )
    {
        res[i]->store(i, force, convergence);
        int t = res[i]->nbLines;
        delete res[i];
        res[i] = NULL;
        i += t;
    }
}


SOFA_DECL_CLASS ( MasterConstraintSolver )

int MasterConstraintSolverClass = core::RegisterObject ( "Constraint solver" )
        .add< MasterConstraintSolver >()
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa
