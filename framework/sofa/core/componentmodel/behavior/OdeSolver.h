#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_ODESOLVER_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_ODESOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class OdeSolver : public objectmodel::BaseObject
{
public:
    typedef BaseMechanicalState::VecId VecId;

    OdeSolver();

    virtual ~OdeSolver();

    virtual void solve (double dt) = 0;

    /// Method called at initialization, during the backwards traversal of the data structure.
    virtual void bwdInit() {}

    /// @name Actions and MultiVectors
    /// This provides an abstract view of the mechanical system to animate
    /// @{

    /// Wait for the completion of previous operations and return the result of the last v_dot call
    virtual double finish();

    virtual VecId v_alloc(VecId::Type t);
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

    virtual void computeMatrix(defaulttype::SofaBaseMatrix *mat=NULL, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0);
    virtual void getMatrixDimension(unsigned int * const, unsigned int * const);
    virtual void computeOpVector(defaulttype::SofaBaseVector *vect=NULL, unsigned int offset=0);
    virtual void matResUpdatePosition(defaulttype::SofaBaseVector *vect=NULL, unsigned int offset=0);

    virtual void computeCompliance(double dt, double **W, double *dFree, int &numContact);

    virtual void print( VecId v, std::ostream& out );
    virtual void printWithElapsedTime( VecId v,  unsigned time );
    /// @}

protected:
    defaulttype::SofaBaseMatrix *mat;

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
    std::map<VecId::Type, VectorIndexAlloc > vectors; ///< Current vectors

    double result;

    class MultiVector
    {
    public:
        typedef OdeSolver::VecId VecId;
    protected:
        core::componentmodel::behavior::OdeSolver* parent;
        VecId v;
        // Copy is forbidden
        MultiVector(const MultiVector& v);
    public:
        MultiVector(core::componentmodel::behavior::OdeSolver* parent, VecId v) : parent(parent), v(v)
        {}
        MultiVector(core::componentmodel::behavior::OdeSolver* parent, VecId::Type t) : parent(parent), v(parent->v_alloc(t))
        {}
        ~MultiVector()
        {
            parent->v_free(v);
        }
        operator VecId()
        {
            return v;
        }

        void clear()
        {
            parent->v_clear(v);
        }

        void eq(VecId a)
        {
            parent->v_eq(v, a);
        }
        void peq(VecId a, double f=1.0)
        {
            parent->v_peq(v, a, f);
        }
        void teq(double f)
        {
            parent->v_teq(v, f);
        }
        double dot(VecId a)
        {
            parent->v_dot(v, a);
            return parent->finish();
        }

        double norm()
        {
            parent->v_dot(v, v);
            return sqrt( parent->finish() );
        }

        void operator=(VecId a)
        {
            eq(a);
        }
        void operator=(const MultiVector& v)
        {
            eq(v.v);
        }
        void operator+=(VecId a)
        {
            peq(a);
        }
        void operator*=(double f)
        {
            teq(f);
        }
        double operator*(VecId a)
        {
            return dot(a);
        }

        friend std::ostream& operator << (std::ostream& out, const MultiVector& mv )
        {
            mv.parent->print(mv.v,out);
            return out;
        }
    };
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif


