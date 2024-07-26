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
#include <sofa/component/odesolver/backward/config.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <fstream>
#include <sofa/core/behavior/LinearSolverAccessor.h>


namespace sofa::component::odesolver::backward
{

/** Implicit and Explicit time integrator using the Variational Symplectic Integrator as defined in :
 * Kharevych, L et al. “Geometric, Variational Integrators for Computer Animation.” ACM SIGGRAPH Symposium on Computer Animation 4 (2006): 43–51.
 *
 * The current implementation for implicit integration assume alpha =0.5 (quadratic accuracy) and uses
 * several Newton steps to estimate the velocity
 *
*/
class SOFA_COMPONENT_ODESOLVER_BACKWARD_API VariationalSymplecticSolver
    : public sofa::core::behavior::OdeSolver
    , public sofa::core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(VariationalSymplecticSolver, sofa::core::behavior::OdeSolver, sofa::core::behavior::LinearSolverAccessor);

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<SReal> f_newtonError;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<unsigned int> f_newtonSteps;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<SReal> f_rayleighStiffness;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<SReal> f_rayleighMass;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<bool> f_saveEnergyInFile;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<bool> f_explicit;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<std::string> f_fileName;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<bool> f_computeHamiltonian;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<SReal> f_hamiltonianEnergy;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<bool> f_useIncrementalPotentialEnergy;

    Data<SReal>       d_newtonError; ///< Error tolerance for Newton iterations
    Data<unsigned int> d_newtonSteps; ///< Maximum number of Newton steps
    Data<SReal> d_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> d_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0
    Data<bool> d_saveEnergyInFile; ///< If kinetic and potential energies should be dumped in a CSV file at each iteration
    Data<bool>       d_explicit; ///< Use explicit integration scheme
    Data<std::string> d_fileName; ///< File name where kinetic and potential energies are saved in a CSV file
    Data<bool> d_computeHamiltonian; ///< Compute hamiltonian
    Data<SReal> d_hamiltonianEnergy; ///< hamiltonian energy
    Data<bool> d_useIncrementalPotentialEnergy; ///< use real potential energy, if false use approximate potential energy
    Data<bool> d_threadSafeVisitor; ///< If true, do not use realloc and free visitors in fwdInteractionForceField.

    SOFA_ATTRIBUTE_DISABLED__ODESOLVER_BACKWARD_VERBOSEDATA()
    sofa::core::objectmodel::lifecycle::RemovedData f_verbose{this, "v23.12", "v24.06", "verbose", "This Data is no longer used"};

    VariationalSymplecticSolver();

    std::ofstream energies;

    void init() override;
    void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    int cpt;

    /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt.
    SReal getVelocityIntegrationFactor() const override
    {
        return 0;
    }

    /// Given a displacement as computed by the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt².
    SReal getPositionIntegrationFactor() const override
    {
        return 0;
    }

    SReal getIntegrationFactor(int /*inputDerivative*/, int /*outputDerivative*/) const override
    {
        return 0;
    }

    SReal getSolutionIntegrationFactor(int /*outputDerivative*/) const override
    {

        return 0;
    }

    void parse(core::objectmodel::BaseObjectDescription *arg) override;

protected:
    sofa::core::MultiVecDerivId pID;
    SReal m_incrementalPotentialEnergy;
};

} // namespace sofa::component::odesolver::backward
