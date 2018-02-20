/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef VariationalSymplecticSolver_H
#define VariationalSymplecticSolver_H
#include "config.h"

#include <sofa/core/behavior/OdeSolver.h>
#include <fstream>

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;

/** Implicit and Explicit time integrator using the Variational Symplectic Integrator as defined in :
 * Kharevych, L et al. “Geometric, Variational Integrators for Computer Animation.” ACM SIGGRAPH Symposium on Computer Animation 4 (2006): 43–51.
 * 
 * The current implementation for implicit integration assume alpha =0.5 (quadratic accuracy) and uses
 * several Newton steps to estimate the velocity
 *
*/
class SOFA_GENERAL_IMPLICIT_ODE_SOLVER_API VariationalSymplecticSolver : public sofa::core::behavior::OdeSolver
{
public:
	SOFA_CLASS(VariationalSymplecticSolver, sofa::core::behavior::OdeSolver);

    Data<double>       f_newtonError; ///< Error tolerance for Newton iterations
    Data<unsigned int> f_newtonSteps; ///< Maximum number of Newton steps
    Data<SReal> f_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> f_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0
	Data<bool> f_verbose; ///< Dump information on the residual errors and number of Newton iterations
    Data<bool> f_saveEnergyInFile; ///< If kinetic and potential energies should be dumped in a CSV file at each iteration
	Data<bool>       f_explicit; ///< Use explicit integration scheme
    Data<std::string> f_fileName; ///< File name where kinetic and potential energies are saved in a CSV file
    Data<bool> f_computeHamiltonian; ///< Compute hamiltonian
    Data<double> f_hamiltonianEnergy; ///< hamiltonian energy
    Data<bool> f_useIncrementalPotentialEnergy; ///< use real potential energy, if false use approximate potential energy

	VariationalSymplecticSolver();

	void init() override;
	std::ofstream energies;
   void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

   int cpt;
   /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
   ///
   /// This method is used to compute the compliance for contact corrections
   /// For Euler methods, it is typically dt.
   virtual double getVelocityIntegrationFactor() const override
   {
       return 0; // getContext()->getDt();
   }

   /// Given a displacement as computed by the linear system inversion, how much will it affect the position
   ///
   /// This method is used to compute the compliance for contact corrections
   /// For Euler methods, it is typically dt².
   virtual double getPositionIntegrationFactor() const override
   {
       return 0; //*getContext()->getDt());
   }


       double getIntegrationFactor(int /*inputDerivative*/, int /*outputDerivative*/) const override
       {

                       return 0;

       }


       double getSolutionIntegrationFactor(int /*outputDerivative*/) const override
       {

                       return 0;
       }
protected:
        sofa::core::MultiVecDerivId pID;
        double m_incrementalPotentialEnergy;


};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
