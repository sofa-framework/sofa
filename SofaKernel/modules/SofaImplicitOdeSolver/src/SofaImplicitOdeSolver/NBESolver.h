#ifndef VNCS_NBESOLVER_H
#define VNCS_NBESOLVER_H

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa::component::odesolver {
class NBESolver : public sofa::core::behavior::OdeSolver {
 public:
  SOFA_CLASS(NBESolver, sofa::core::behavior::OdeSolver);
  NBESolver();
  void init() override;

  void cleanup() override;

  void solve(const sofa::core::ExecParams *params, SReal dt,
             sofa::core::MultiVecCoordId xResult,
             sofa::core::MultiVecDerivId vResult) override;

  SReal rayleighStiffness() const;
  void setRayleighStiffness(SReal newRayleighStiffness);

  SReal rayleighMass() const;
  void setRayleighMass(SReal newRayleighMass);

  SReal newtonThreshold() const;
  void setNewtonThreshold(SReal newNewtonThreshold);

  int maxNewtonIterations() const;
  void setMaxNewtonIterations(int newMaxNewtonIterations);

  int lineSearchMaxIterations() const;
  void setLineSearchMaxIterations(int newLineSearchMaxIterations);

 private:
  SReal m_rayleighStiffness;
  SReal m_rayleighMass;
  SReal m_newtonThreshold;
  int m_maxNewtonIterations;
  int m_lineSearchMaxIterations;
};
}  // namespace sofa::component::odesolver

#endif  // VNCS_NBESOLVER_H
