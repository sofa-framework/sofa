#ifndef SOFA_CORE_ODESOLVER_H
#define SOFA_CORE_ODESOLVER_H

#include "Sofa/Abstract/BaseObject.h"
#include "Sofa/Components/Common/SofaBaseMatrix.h"
#include "Encoding.h"

namespace Sofa
{

namespace Core
{

class OdeSolver : public Abstract::BaseObject
{
public:
    OdeSolver();

    virtual ~OdeSolver();

    virtual void solve (double dt) = 0;

protected:
    Components::Common::SofaBaseMatrix *mat;

    /// @name Actions and MultiVectors
    /// This provides an abstract view of the mechanical system to animate
    /// @{


protected:

    typedef Encoding::VecId VecId;
    class VectorIndexAlloc
    {
    protected:
        std::set<unsigned int> vused; ///< Currently in-use vectors
        std::set<unsigned int> vfree; ///< Once used vectors
        unsigned int  maxIndex; ///< Max index used
    public:
        VectorIndexAlloc();
        unsigned int alloc();
        bool free(unsigned int v);
    };
    std::map<Core::Encoding::VecType, VectorIndexAlloc > vectors; ///< Current vectors

    double result;
public:
    /// Wait for the completion of previous operations and return the result of the last v_dot call
    virtual double finish();

    virtual VecId v_alloc(Core::Encoding::VecType t);
    virtual void v_free(VecId v);

    virtual void v_clear(VecId v); ///< v=0
    virtual void v_eq(VecId v, VecId a); ///< v=a
    virtual void v_peq(VecId v, VecId a, double f=1.0); ///< v+=f*a
    virtual void v_teq(VecId v, double f); ///< v*=f
    virtual void v_dot(VecId a, VecId b); ///< a dot b ( get result using finish )
    virtual void propagateDx(VecId dx);
    virtual void projectResponse(VecId dx);
    virtual void addMdx(VecId res, VecId dx);
    virtual void integrateVelocity(VecId res, VecId x, VecId v, double dt);
    virtual void accFromF(VecId a, VecId f);
    virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    virtual void computeForce(VecId result);
    virtual void computeDf(VecId df);
    virtual void computeAcc(double t, VecId a, VecId x, VecId v);

    virtual void computeMatrix(Components::Common::SofaBaseMatrix *mat=NULL, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0);
    virtual void getMatrixDimension(unsigned int * const, unsigned int * const);
    virtual void computeOpVector(Components::Common::SofaBaseVector *vect=NULL, unsigned int offset=0);
    virtual void matResUpdatePosition(Components::Common::SofaBaseVector *vect=NULL, unsigned int offset=0);

    virtual void print( VecId v, std::ostream& out );
    /// @}
};

} // namespace Core

} // namespace Sofa

#endif


