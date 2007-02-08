#ifndef SOFA_HELPER_LCPSOLVER_H
#define SOFA_HELPER_LCPSOLVER_H

namespace sofa
{

namespace helper
{

template <int dim>
class LCPSolver
{
public:
    typedef double Matrix[dim][dim];

    bool  solve(const double *q, const Matrix &M, double *res);
    void  printInfo(double *q, Matrix &M);
};

} // namespace helper

} // namespace sofa

#endif
