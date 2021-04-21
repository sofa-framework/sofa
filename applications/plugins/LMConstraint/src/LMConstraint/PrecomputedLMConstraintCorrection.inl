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
#pragma once
#include <LMConstraint/PrecomputedLMConstraintCorrection.h>
#include <SofaConstraint/PrecomputedConstraintCorrection.inl>
#include <LMConstraint/LMConstraintSolver.h>


namespace sofa::component::constraintset
{

template<class DataTypes>
void PrecomputedLMConstraintCorrection<DataTypes>::bwdInit()
{
    Inherit::init();

    const VecDeriv& v0 = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();

    this->nbNodes = v0.size();

    if (this->nbNodes == 0)
    {
        msg_error() << "No degree of freedom" ;
        return;
    }

    this->dof_on_node = v0[0].size();

    this->nbRows = this->nbNodes * this->dof_on_node;
    this->nbCols = this->nbNodes * this->dof_on_node;

    double dt = this->getContext()->getDt();

    this->invName = f_fileCompliance.getFullPath().empty() ? this->buildFileName() : f_fileCompliance.getFullPath();

    if (!this->loadCompliance(this->invName))
    {
        msg_info() << "Compliance being built";

        // Buffer Allocation
        this->invM->data = new Real[this->nbRows * this->nbCols];

        // for the intial computation, the gravity has to be put at 0
        const sofa::defaulttype::Vec3d gravity = this->getContext()->getGravity();

        const sofa::defaulttype::Vec3d gravity_zero(0.0,0.0,0.0);
        this->getContext()->setGravity(gravity_zero);

        sofa::component::odesolver::EulerImplicitSolver* eulerSolver;
        sofa::component::linearsolver::CGLinearSolver< sofa::component::linearsolver::GraphScatteredMatrix, sofa::component::linearsolver::GraphScatteredVector >* cgLinearSolver;
        core::behavior::LinearSolver* linearSolver;

        this->getContext()->get(eulerSolver);
        this->getContext()->get(cgLinearSolver);
        this->getContext()->get(linearSolver);

        simulation::Node *solvernode = nullptr;

        if (eulerSolver && cgLinearSolver)
        {
            msg_info() << "use EulerImplicitSolver & CGLinearSolver" ;
            solvernode = (simulation::Node*)eulerSolver->getContext();
        }
        else if (eulerSolver && linearSolver)
        {
            msg_info() << "use EulerImplicitSolver & LinearSolver";
            solvernode = (simulation::Node*)eulerSolver->getContext();
        }
        else if(eulerSolver)
        {
            msg_info() << "use EulerImplicitSolver";
            solvernode = (simulation::Node*)eulerSolver->getContext();
        }
        else
        {
            msg_error() << "PrecomputedContactCorrection must be associated with EulerImplicitSolver+LinearSolver for the precomputation\nNo Precomputation" ;
            return;
        }

        helper::vector< sofa::component::constraintset::LMConstraintSolver* > listLMConstraintSolver;
        solvernode->get< sofa::component::constraintset::LMConstraintSolver >(&listLMConstraintSolver, core::objectmodel::BaseContext::SearchDown);
        helper::vector< sofa::component::constraintset::ConstraintActivation > listConstraintActivation(listLMConstraintSolver.size());
        for (unsigned int i=0; i<listLMConstraintSolver.size(); ++i)
        {
            listConstraintActivation[i].acc=listLMConstraintSolver[i]->constraintAcc.getValue();
            listLMConstraintSolver[i]->constraintAcc.setValue(false);

            listConstraintActivation[i].vel=listLMConstraintSolver[i]->constraintVel.getValue();
            listLMConstraintSolver[i]->constraintVel.setValue(false);

            listConstraintActivation[i].pos=listLMConstraintSolver[i]->constraintPos.getValue();
            listLMConstraintSolver[i]->constraintPos.setValue(false);
        }

        // Change the solver parameters
        double buf_tolerance = 0, buf_threshold = 0;
        int	   buf_maxIter = 0;

        if (cgLinearSolver)
        {
            buf_tolerance = (double) cgLinearSolver->f_tolerance.getValue();
            buf_maxIter   = (int) cgLinearSolver->f_maxIter.getValue();
            buf_threshold = (double) cgLinearSolver->f_smallDenominatorThreshold.getValue();

            cgLinearSolver->f_tolerance.setValue(1e-20);
            cgLinearSolver->f_maxIter.setValue(5000);
            cgLinearSolver->f_smallDenominatorThreshold.setValue(1e-35);
        }


        helper::ReadAccessor< Data< VecCoord > > rposData = *this->mstate->read(core::ConstVecCoordId::position());
        const VecCoord prev_pos = rposData.ref();

        helper::WriteAccessor< Data< VecDeriv > > velocityData = *this->mstate->write(core::VecDerivId::velocity());
        VecDeriv& velocity = velocityData.wref();
        const VecDeriv prev_velocity = velocity;

        helper::WriteAccessor< Data< VecDeriv > > forceData = *this->mstate->write(core::VecDerivId::externalForce());
        VecDeriv& force = forceData.wref();
        force.clear();
        force.resize(this->nbNodes);

        /// christian : it seems necessary to called the integration one time for initialization
        /// (avoid to have a line of 0 at the top of the matrix)
        if (eulerSolver)
        {
            eulerSolver->solve(core::execparams::defaultInstance(), dt, core::VecCoordId::position(), core::VecDerivId::velocity());
        }

        Deriv unitary_force;

        std::stringstream tmpStr;
        for (unsigned int f = 0; f < this->nbNodes; f++)
        {
            std::streamsize prevPrecision = sout.precision();
            tmpStr.precision(2);
            tmpStr << "Precomputing constraint correction : " << std::fixed << (float)f / (float)this->nbNodes * 100.0f << " %   " << '\xd';
            sout.precision(prevPrecision);

            for (unsigned int i = 0; i < this->dof_on_node; i++)
            {
                unitary_force.clear();
                unitary_force[i] = 1.0;

                force[f] = unitary_force;

                // Reset positions and velocities
                velocity.clear();
                velocity.resize(this->nbNodes);

                // Actualize ref to the position vector ; it seems it is changed at every eulerSolver->solve()
                helper::WriteOnlyAccessor< Data< VecCoord > > wposData = *this->mstate->write(core::VecCoordId::position());
                VecCoord& pos = wposData.wref();

                for (unsigned int n = 0; n < this->nbNodes; n++)
                    pos[n] = prev_pos[n];

                double fact = 1.0 / dt; // christian : it is not a compliance... but an admittance that is computed !

                if (eulerSolver)
                {
                    fact *= eulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

                    eulerSolver->solve(core::execparams::defaultInstance(), dt, core::VecCoordId::position(), core::VecDerivId::velocity());

                    if (linearSolver)
                        linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation
                }

                for (unsigned int v = 0; v < this->nbNodes; v++)
                {
                    for (unsigned int j = 0; j < this->dof_on_node; j++)
                    {
                        this->invM->data[(v * this->dof_on_node + j) * this->nbCols + (f * this->dof_on_node + i) ] = (Real)(fact * velocity[v][j]);
                    }
                }
            }

            unitary_force.clear();
            force[f] = unitary_force;
        }
        msg_info() << tmpStr.str();

        // Do not recompute the matrix for the rest of the precomputation
        if (linearSolver)
            linearSolver->freezeSystemMatrix();

        this->saveCompliance(this->invName);

        // Restore gravity
        this->getContext()->setGravity(gravity);

        // Restore linear solver parameters
        if (cgLinearSolver)
        {
            cgLinearSolver->f_tolerance.setValue(buf_tolerance);
            cgLinearSolver->f_maxIter.setValue(buf_maxIter);
            cgLinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
        }

        // Retore velocity
        for (unsigned int i = 0; i < velocity.size(); i++)
            velocity[i] = prev_velocity[i];

        helper::WriteOnlyAccessor< Data< VecCoord > > wposData = *this->mstate->write(core::VecCoordId::position());
        VecCoord& pos = wposData.wref();

        // Restore position
        for (unsigned int i = 0; i < pos.size(); i++)
            pos[i] = prev_pos[i];

        for (unsigned int i=0; i<listLMConstraintSolver.size(); ++i)
        {
            listLMConstraintSolver[i]->constraintAcc.setValue(listConstraintActivation[i].acc);
            listLMConstraintSolver[i]->constraintVel.setValue(listConstraintActivation[i].vel);
            listLMConstraintSolver[i]->constraintPos.setValue(listConstraintActivation[i].pos);
        }
    }

    this->appCompliance = this->invM->data;

    // Optimisation for the computation of W
    this->_indexNodeSparseCompliance.resize(v0.size());

    //  Print 400 first row and column of the matrix
    if (this->notMuted())
    {
        msg_info() << "Matrix compliance : this->nbCols = " << this->nbCols << "  this->nbRows =" << this->nbRows;

        for (unsigned int i = 0; i < 20 && i < this->nbCols; i++)
        {
            for (unsigned int j = 0; j < 20 && j < this->nbCols; j++)
            {
                msg_info() << " \t " << this->appCompliance[j*this->nbCols + i];
            }
        }

    }
}


}
