#ifndef SOFA_COMPONENTS_PROXIMITY_LCP_SOLVER_H
#define SOFA_COMPONENTS_PROXIMITY_LCP_SOLVER_H

namespace Sofa
{

namespace Components
{

namespace Proximity
{

template <int dim>
class LCP
{
public:
    typedef double Matrix[dim][dim];

    bool  solve(const double *q, const Matrix &M, double *res);
    void  printInfo(double *q, Matrix &M);
};

} // namespace Proximity

} // namespace Components

} // namespace Sofa

#endif
