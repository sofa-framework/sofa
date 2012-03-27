#ifndef SOFA_COMPONENT_ODESOLVER_MinresSolver_H
#define SOFA_COMPONENT_ODESOLVER_MinresSolver_H

#include "ComplianceSolver.h"

namespace sofa
{
using helper::vector;

namespace component
{

namespace odesolver
{

/** Solver using a regularized KKT matrix.
  The compliance values are used to regularize the system.


  Inspired from Servin,Lacoursière,Melin, Interactive Simulation of Elastic Deformable Materials,  http://www.ep.liu.se/ecp/019/005/ecp01905.pdf

  We generalize it to a tunable implicit integration scheme:
      \f[ \begin{array}{ccc}
    \Delta v &=& h.M^{-1}.(\alpha f_{n+1} + (1-\alpha) f_n)  \\
    \Delta x &=& h.(\beta v_{n+1} + (1-\beta) v_n)
    \end{array} \f]
    where \f$ h \f$ is the time step, \f$ \alpha \f$ is the implicit velocity factor, and \f$ \beta \f$ is the implicit position factor.

    The corresponding dynamic equation is:
  \f[ \left( \begin{array}{cc} M & -J^T \\
                               J & \frac{1}{l} C \end{array}\right)
      \left( \begin{array}{c} \delta v \\ \bar\lambda \end{array}\right)
    = \left( \begin{array}{c} f \\ - \frac{1}{l} (\phi +(d+\alpha h) \dot \phi)  \end{array}\right) \f]
    where \f$ M \f$ is the mass matrix, \f$ \phi \f$ is the constraint violation, \f$ J \f$ the constraint Jacobian matrix,
    \f$ C \f$ is the compliance matrix (i.e. inverse of constraint stiffness), \f$ l=\alpha(h \beta + d) \f$ is a term related to implicit integration and constraint damping, and
      \f$ \bar\lambda \f$ is the average constraint forces, consistently with the implicit velocity integration.

  A Shur complement is used to compute the constraint forces, then these are added to the external forces to obtain the final velocity increment,
  and the positions are updated according to the implicit scheme:

  \f[ \begin{array}{ccc}
   ( JPM^{-1}PJ^T + \frac{1}{l}C ) \bar\lambda &=& \frac{-1}{l} (\phi + (d+h\alpha)\dot\phi ) - J M^{-1} f \\
                                 \Delta v  &=&  P M^{-1}( f + J^T \bar\lambda ) \\
                                 \Delta x  &=&  h( v + \beta \Delta v )
  \end{array} \f]
where \f$ P \f$ is the projection matrix corresponding to the projective constraints applied to the independent DOFs.

\sa \ref sofa::core::behavior::BaseCompliance



*/
class SOFA_Compliant_API MinresSolver  : public ComplianceSolver
{
public:
    SOFA_CLASS(MinresSolver, sofa::core::behavior::OdeSolver);

    virtual void solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);




};

}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
