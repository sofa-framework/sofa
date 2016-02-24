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
class SOFA_IMPLICIT_ODE_SOLVER_API VariationalSymplecticSolver : public sofa::core::behavior::OdeSolver
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

	void init();
	std::ofstream energies;
   void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);

   int cpt;
   /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
   ///
   /// This method is used to compute the compliance for contact corrections
   /// For Euler methods, it is typically dt.
   virtual double getVelocityIntegrationFactor() const
   {
       return 0; // getContext()->getDt();
   }

   /// Given a displacement as computed by the linear system inversion, how much will it affect the position
   ///
   /// This method is used to compute the compliance for contact corrections
   /// For Euler methods, it is typically dt².
   virtual double getPositionIntegrationFactor() const
   {
       return 0; //*getContext()->getDt());
   }


       double getIntegrationFactor(int /*inputDerivative*/, int /*outputDerivative*/) const
       {

                       return 0;

       }


       double getSolutionIntegrationFactor(int /*outputDerivative*/) const
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
