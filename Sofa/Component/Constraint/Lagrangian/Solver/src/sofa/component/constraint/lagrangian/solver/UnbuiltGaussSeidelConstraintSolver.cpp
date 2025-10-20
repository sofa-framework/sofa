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

#include <sofa/component/constraint/lagrangian/solver/UnbuiltGaussSeidelConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/UnbuiltConstraintProblem.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::component::constraint::lagrangian::solver
{


void UnbuiltGaussSeidelConstraintSolver::doSolve(GenericConstraintProblem * problem , SReal timeout)
{
    UnbuiltConstraintProblem* c_current_cp = dynamic_cast<UnbuiltConstraintProblem*>(problem);
    if (c_current_cp == nullptr)
    {
        msg_error()<<"Constraint problem must derive from UnbuiltConstraintProblem";
        return;
    }

    SCOPED_TIMER_VARNAME(unbuiltGaussSeidelTimer, "ConstraintsUnbuiltGaussSeidel");


    if(!c_current_cp->getDimension())
    {
        c_current_cp->currentError = 0.0;
        c_current_cp->currentIterations = 0;
        return;
    }

    SReal t0 = (SReal)sofa::helper::system::thread::CTime::getTime();
    SReal timeScale = 1.0 / (SReal)sofa::helper::system::thread::CTime::getTicksPerSec();

    SReal *dfree = c_current_cp->getDfree();
    SReal *force = c_current_cp->getF();
    SReal **w = c_current_cp->getW();
    SReal tol = c_current_cp->tolerance;

    SReal *d = c_current_cp->_d.ptr();

    unsigned int iter = 0, nb = 0;

    SReal error=0.0;

    bool convergence = false;
    sofa::type::vector<SReal> tempForces;
    if(c_current_cp->sor != 1.0) tempForces.resize(c_current_cp->getDimension());

    if(c_current_cp->scaleTolerance && !c_current_cp->allVerified)
        tol *= c_current_cp->getDimension();


    for(int i=0; i<c_current_cp->getDimension(); )
    {
        if(!c_current_cp->constraintsResolutions[i])
        {
            msg_warning() << "Bad size of constraintsResolutions in GenericConstraintSolver" ;
            c_current_cp->setDimension(i);
            break;
        }
        c_current_cp->constraintsResolutions[i]->init(i, w, force);
        i += c_current_cp->constraintsResolutions[i]->getNbLines();
    }
    memset(force, 0, c_current_cp->getDimension() * sizeof(SReal));	// Erase previous forces for the time being


    bool showGraphs = false;
    sofa::type::vector<SReal>* graph_residuals = nullptr;
    std::map < std::string, sofa::type::vector<SReal> > *graph_forces = nullptr, *graph_violations = nullptr;
    sofa::type::vector<SReal> tabErrors;


    showGraphs = d_computeGraphs.getValue();

    if(showGraphs)
    {
        graph_forces = d_graphForces.beginEdit();
        graph_forces->clear();

        graph_violations = d_graphViolations.beginEdit();
        graph_violations->clear();

        graph_residuals = &(*d_graphErrors.beginEdit())["Error"];
        graph_residuals->clear();
    }

    tabErrors.resize(c_current_cp->getDimension());

    // temporary buffers
    std::vector<SReal> errF;
    std::vector<SReal> tempF;

    for(iter=0; iter < static_cast<unsigned int>(c_current_cp->maxIterations); iter++)
    {
        bool constraintsAreVerified = true;
        if(c_current_cp->sor != 1.0)
        {
            std::copy_n(force, c_current_cp->getDimension(), tempForces.begin());
        }

        error=0.0;
        for (auto it_c = c_current_cp->constraints_sequence.begin(); it_c != c_current_cp->constraints_sequence.end(); )  // increment of it_c realized at the end of the loop
        {
            const auto j = *it_c;
            //1. nbLines provide the dimension of the constraint
            nb = c_current_cp->constraintsResolutions[j]->getNbLines();

            //2. for each line we compute the actual value of d
            //   (a)d is set to dfree
            if(nb > errF.size())
            {
                errF.resize(nb);
            }
            std::copy_n(&force[j], nb, errF.begin());
            std::copy_n(&dfree[j], nb, &d[j]);

            //   (b) contribution of forces are added to d
            for (auto* el : c_current_cp->cclist_elems[j])
            {
                if (el)
                    el->addConstraintDisplacement(d, j, j+nb-1);
            }

            //3. the specific resolution of the constraint(s) is called
            c_current_cp->constraintsResolutions[j]->resolution(j, w, d, force, dfree);

            //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
            SReal contraintError = 0.0;
            if(nb > 1)
            {
                for(unsigned int l=0; l<nb; l++)
                {
                    SReal lineError = 0.0;
                    for (unsigned int m=0; m<nb; m++)
                    {
                        SReal dofError = w[j+l][j+m] * (force[j+m] - errF[m]);
                        lineError += dofError * dofError;
                    }
                    lineError = sqrt(lineError);
                    if(lineError > tol)
                        constraintsAreVerified = false;

                    contraintError += lineError;
                }
            }
            else
            {
                contraintError = fabs(w[j][j] * (force[j] - errF[0]));
                if(contraintError > tol)
                    constraintsAreVerified = false;
            }

            if(c_current_cp->constraintsResolutions[j]->getTolerance())
            {
                if(contraintError > c_current_cp->constraintsResolutions[j]->getTolerance())
                    constraintsAreVerified = false;
                contraintError *= tol / c_current_cp->constraintsResolutions[j]->getTolerance();
            }

            error += contraintError;
            tabErrors[j] = contraintError;

            //5. the force is updated for the constraint corrections
            bool update = false;
            for(unsigned int l=0; l<nb; l++)
                update |= (force[j+l] || errF[l]);

            if(update)
            {
                if (nb > tempF.size())
                {
                    tempF.resize(nb);
                }
                std::copy_n(&force[j], nb, tempF.begin());
                for(unsigned int l=0; l<nb; l++)
                {
                    force[j+l] -= errF[l]; // DForce
                }

                for (auto* el : c_current_cp->cclist_elems[j])
                {
                    if (el)
                        el->setConstraintDForce(force, j, j+nb-1, update);
                }

                std::copy_n(tempF.begin(), nb, &force[j]);
            }
            std::advance(it_c, nb);
        }

        if(showGraphs)
        {
            for(int j=0; j<c_current_cp->getDimension(); j++)
            {
                std::ostringstream oss;
                oss << "f" << j;

                sofa::type::vector<SReal>& graph_force = (*graph_forces)[oss.str()];
                graph_force.push_back(force[j]);

                sofa::type::vector<SReal>& graph_violation = (*graph_violations)[oss.str()];
                graph_violation.push_back(d[j]);
            }

            graph_residuals->push_back(error);
        }

        if(c_current_cp->sor != 1.0)
        {
            for(int j=0; j<c_current_cp->getDimension(); j++)
                force[j] = c_current_cp->sor * force[j] + (1-c_current_cp->sor) * tempForces[j];
        }
        if(timeout)
        {
            SReal t1 = (SReal)sofa::helper::system::thread::CTime::getTime();
            SReal dt = (t1 - t0)*timeScale;

            if(dt > timeout)
            {
                c_current_cp->currentError = error;
                c_current_cp->currentIterations = iter+1;
                return;
            }
        }
        else if(c_current_cp->allVerified)
        {
            if(constraintsAreVerified)
            {
                convergence = true;
                break;
            }
        }
        else if(error < tol)
        {
            convergence = true;
            break;
        }
    }



    sofa::helper::AdvancedTimer::valSet("GS iterations", c_current_cp->currentIterations);

    c_current_cp->result_output(this, force, error, iter, convergence);

    if(showGraphs)
    {
        d_graphErrors.endEdit();

        sofa::type::vector<SReal>& graph_constraints = (*d_graphConstraints.beginEdit())["Constraints"];
        graph_constraints.clear();

        for(int j=0; j<c_current_cp->getDimension(); )
        {
            nb = c_current_cp->constraintsResolutions[j]->getNbLines();

            if(tabErrors[j])
                graph_constraints.push_back(tabErrors[j]);
            else if(c_current_cp->constraintsResolutions[j]->getTolerance())
                graph_constraints.push_back(c_current_cp->constraintsResolutions[j]->getTolerance());
            else
                graph_constraints.push_back(tol);

            j += nb;
        }
        d_graphConstraints.endEdit();

        d_graphForces.endEdit();
    }
}

void registerUnbuiltGaussSeidelConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using an Unbuilt version of the Gauss-Seidel iterative method")
        .add< UnbuiltGaussSeidelConstraintSolver >());
}

}