#ifndef __LCP_SOLVER__
#define __LCP_SOLVER__


template <int dim> class LCP
{


public:

    typedef double Matrix[dim][dim];

    bool  solve(const double *q, const Matrix &M, double *res);
    void  printInfo(double *q, Matrix &M);

};

#endif // __LCP_SOLVER__
