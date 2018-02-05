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

    Data<double>       f_newtonError;
    Data<unsigned int> f_newtonSteps;
    Data<SReal> f_rayleighStiffness;
    Data<SReal> f_rayleighMass;
	Data<bool> f_verbose;
    Data<bool> f_saveEnergyInFile;
	Data<bool>       f_explicit;
    Data<std::string> f_fileName;
    Data<bool> f_computeHamiltonian;
    Data<double> f_hamiltonianEnergy;
    Data<bool> f_useIncrementalPotentialEnergy;

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
