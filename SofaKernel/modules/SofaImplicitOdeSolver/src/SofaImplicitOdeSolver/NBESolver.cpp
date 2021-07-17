#include "NBESolver.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

namespace sofa::component::odesolver {
using sofa::core::VecId;
using namespace sofa::defaulttype;
using namespace sofa::core::behavior;

NBESolver::NBESolver()
    : sofa::core::behavior::OdeSolver(),
      m_rayleighStiffness(0),
      m_rayleighMass(0),
      m_newtonThreshold(1e-5),
      m_maxNewtonIterations(1),
      m_lineSearchMaxIterations(1) {}

void NBESolver::init() { sofa::core::behavior::OdeSolver::init(); }

void NBESolver::cleanup() {}

void NBESolver::solve(const sofa::core::ExecParams *params, SReal dt,
                      sofa::core::MultiVecCoordId xResult,
                      sofa::core::MultiVecDerivId vResult) {
  sofa::simulation::common::VectorOperations vop(params, this->getContext());
  sofa::simulation::common::MechanicalOperations mop(params,
                                                     this->getContext());
  MultiVecCoord pos(&vop, sofa::core::VecCoordId::position());
  MultiVecDeriv vel(&vop, sofa::core::VecDerivId::velocity());
  MultiVecDeriv f(&vop, sofa::core::VecDerivId::force());
  MultiVecDeriv b(&vop);
  MultiVecCoord newPos(&vop, xResult);
  MultiVecDeriv newVel(&vop, vResult);

  MultiVecCoord pos_i1(&vop);
  MultiVecDeriv vel_i1(&vop);

  // This two will be used to test the line search current approximation
  // They will contain the final result of the simulation
  pos_i1.realloc(&vop, false, true);
  vel_i1.realloc(&vop, false, true);

  MultiVecCoord pos0(&vop);
  MultiVecDeriv vel0(&vop);
  pos0.realloc(&vop, false, true);
  vel0.realloc(&vop, false, true);

  /// inform the constraint parameters about the position and velocity id
  mop.cparams.setX(xResult);
  mop.cparams.setV(vResult);

  // dx is no longer allocated by default (but it will be deleted automatically
  // by the mechanical objects)
  MultiVecDeriv dx(&vop, sofa::core::VecDerivId::dx());
  dx.realloc(&vop, false, true);

  const SReal &h = dt;

  sofa::helper::AdvancedTimer::stepBegin("ComputeForce");
  mop->setImplicit(true);  // this solver is implicit

  // Here starts my shit
  // Copy current position and velocity to use it later

  vop.v_eq(pos0, pos);
  vop.v_eq(vel0, vel);

  // Perform an explicit step
  vop.v_peq(pos, vel, dt);

  // Propagate explicit step
  mop.propagateX(pos);
  mop.propagateV(vel);

  f.eq(vel0);                               // f = v_0
  f.peq(vel, -(1.0 + h * m_rayleighMass));  // f = v_0 - (1+h*alpha)*v
  b.clear();                                // b = 0
  mop.addMdx(b, f, 1.0);                    // b = M * (v_0 - (1+h*alpha)*v)
  mop.computeForce(f);                      // f = FORCES
  b.peq(f, h);  // b = h*f + M (v_0 - (1+h*alpha * v_i))
  mop.addMBKv(b, 0.0, 0.0,
              h * m_rayleighStiffness);  // b = h * f + M(v_0 - (1+h*alpha)) +
                                         // h*beta*K*v
  mop.projectResponse(b);

  SReal error = b.dot(b);
  int currentNewtonIteration = 0;

  do {
    sofa::core::behavior::MultiMatrix<
        sofa::simulation::common::MechanicalOperations>
        matrix(&mop);
    matrix = MechanicalMatrix(
        1.0 + h * m_rayleighMass,         // mass component
        -h,                               // damping component
        -h * h - m_rayleighStiffness * h  // stiffness component
    );

    mop->setDx(dx);
    matrix.solve(dx, b);  // it says dx, but its a change in the velocities!

    SReal lineSearchStep = 1.0;
    int lineSearchCurrentIteration = 0;
    SReal error_i1 = 0.0;
    bool linearSolveIsSucessful = false;

    // Perform a line search on the direction of dx
    do {
      vel_i1.eq(vel);
      pos_i1.eq(pos0);

      vel_i1.peq(dx, lineSearchStep);  // v_i1 += step * v
      pos_i1.peq(vel_i1, h);           // pos_i1 += h * v_i1

      // Propagate this temptative position and velocity
      mop.propagateXAndV(pos_i1, vel_i1);

      // Check if in this situation, we have less error for the original
      // equation we are solving
      f.eq(vel0);                               // f = v_0
      f.peq(vel, -(1.0 + h * m_rayleighMass));  // f = v_0 - (1+h*alpha)*v
      b.clear();                                // b = 0
      mop.addMdx(b, f, 1.0);                    // b = M * (v_0 - (1+h*alpha)*v)
      mop.computeForce(f);                      // f = FORCES
      b.peq(f, h);  // b = h*f + M (v_0 - (1+h*alpha * v_i))
      mop.addMBKv(b, 0.0, 0.0,
                  h * m_rayleighStiffness);  // b = h * f + M(v_0 - (1+h*alpha))
                                             // + h*beta*K*v
      mop.projectResponse(b);

      error_i1 = b.dot(b);

      if (error_i1 < error) {
        pos.eq(pos_i1);
        vel.eq(vel_i1);

        linearSolveIsSucessful = true;

        break;
      }

      lineSearchStep *= 0.5;
      lineSearchCurrentIteration++;
    } while (lineSearchCurrentIteration < m_lineSearchMaxIterations);

    error = error_i1;

    if (!linearSolveIsSucessful) {
      // The lineSearch didn't converge. This means the search direction was
      // really bad and it doesn't improve current result. Return current
      // result, but the simulation has failed!
      mop.propagateXAndV(pos0, vel0);
    }
    currentNewtonIteration++;
  } while (error > m_newtonThreshold &&
           currentNewtonIteration < m_maxNewtonIterations);

  if (error > m_newtonThreshold) {
    // The newton solver has not reduce the error as much as desired in the
    // available newton iterations. Return current result, but the simulation
    // has failed!
    mop.propagateXAndV(pos0, vel0);
  }

  newPos.eq(pos);
  newVel.eq(vel);
}

SReal NBESolver::rayleighStiffness() const { return m_rayleighStiffness; }

void NBESolver::setRayleighStiffness(SReal newRayleighStiffness) {
  m_rayleighStiffness = newRayleighStiffness;
}

SReal NBESolver::rayleighMass() const { return m_rayleighMass; }

void NBESolver::setRayleighMass(SReal newRayleighMass) {
  m_rayleighMass = newRayleighMass;
}

SReal NBESolver::newtonThreshold() const { return m_newtonThreshold; }

void NBESolver::setNewtonThreshold(SReal newNewtonThreshold) {
  m_newtonThreshold = newNewtonThreshold;
}

int NBESolver::maxNewtonIterations() const { return m_maxNewtonIterations; }

void NBESolver::setMaxNewtonIterations(int newMaxNewtonIterations) {
  m_maxNewtonIterations = newMaxNewtonIterations;
}

int NBESolver::lineSearchMaxIterations() const {
  return m_lineSearchMaxIterations;
}

void NBESolver::setLineSearchMaxIterations(int newLineSearchMaxIterations) {
  m_lineSearchMaxIterations = newLineSearchMaxIterations;
}

}  // namespace sofa::component::odesolver
