#ifndef COMPLIANTPOSTSTABILIZATIONANIMATIONLOOP_H
#define COMPLIANTPOSTSTABILIZATIONANIMATIONLOOP_H



#include <Compliant/config.h>
#include <sofa/simulation/CollisionAnimationLoop.h>


namespace sofa
{

namespace component
{

// forward declarations
namespace odesolver {
    class CompliantImplicitSolver;
}
namespace collision {
    class DefaultContactManager;
}

namespace animationloop
{


/** Implementation of "Post-Stabilization for Rigid Body Simulation with Contact and Constraints", Cline & Pai, ICRA 2003
 * with a full contact update before the correction pass.
 * Note a cheaper but less accurate post-stab without a full collision detection before the correction pass can be obtained with the Data 'stabilization' of CompliantImplicitSolver
 *
 * @author Matthieu Nesme
 * @date 2014
*/
class SOFA_Compliant_API CompliantPostStabilizationAnimationLoop : public sofa::simulation::CollisionAnimationLoop
{
public:

    typedef sofa::simulation::CollisionAnimationLoop Inherit;

    SOFA_CLASS(CompliantPostStabilizationAnimationLoop, sofa::simulation::CollisionAnimationLoop);

protected:

    CompliantPostStabilizationAnimationLoop(simulation::Node* gnode);

    ~CompliantPostStabilizationAnimationLoop() override {}

public:


    void init() override;

    void step( const sofa::core::ExecParams* params, SReal dt ) override;


protected :

    odesolver::CompliantImplicitSolver *m_solver; ///< to ba able to call the post-stab
    collision::DefaultContactManager   *m_contact; ///< keep an eye on the contact manager to change its parameters

    unsigned int m_responseId, m_correctionResponseId; ///< reponse params for the user selected method
    std::string m_responseParams, m_correctionResponseParams; ///< enforces response params for the correction pass (unilateral contact only)

};

} // namespace animationloop

} // namespace component

} // namespace sofa

#endif
