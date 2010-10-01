#ifndef SOFA_SIMULATION_COMMON_MECHANICALOPERATIONS_H
#define SOFA_SIMULATION_COMMON_MECHANICALOPERATIONS_H

#include <sofa/simulation/common/common.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/VecId.h>


namespace sofa
{
namespace core
{
class MechanicalParams;
namespace objectmodel
{
class Context;
}
}
namespace simulation
{
namespace common
{


class SOFA_SIMULATION_COMMON_API MechanicalOperations
{
protected:
    core::MechanicalParams* mparams;
    core::objectmodel::Context* ctx;
public:

    MechanicalOperations(core::MechanicalParams* mparams, core::objectmodel::Context* ctx);
    /// Propagate the given displacement through all mappings
    void propagateDx(core::MultiVecDerivId dx) {};
    /// Propagate the given displacement through all mappings and reset the current force delta
    void propagateDxAndResetDf(core::MultiVecDerivId dx, core::MultiVecDerivId df) {};
    /// Propagate the given position through all mappings
    void propagateX(core::MultiVecCoordId x) {};
    /// Propagate the given position through all mappings and reset the current force delta
    void propagateXAndResetF(core::MultiVecCoordId x, core::MultiVecDerivId f) {};
    /// Apply projective constraints to the given vector
    void projectResponse(core::MultiVecDerivId dx, double **W=NULL) {};
    void addMdx(core::MultiVecDerivId res, core::MultiVecDerivId dx(core::VecDerivId() ) , double factor = 1.0) {}; ///< res += factor M.dx
    void integrateVelocity(core::MultiVecDerivId res, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v, double dt) {}; ///< res = x + v.dt
    void accFromF(core::MultiVecDerivId a, core::MultiVecDerivId f); ///< a = M^-1 . f

    /// Compute the current force (given the latest propagated position and velocity)
    void computeForce(core::MultiVecDerivId result, bool clear = true, bool accumulate = true) {};
    /// Compute the current force delta (given the latest propagated displacement)
    void computeDf(core::MultiVecDerivId df, bool clear = true, bool accumulate = true) {};
    /// Compute the current force delta (given the latest propagated velocity)
    void computeDfV(core::MultiVecDerivId df, bool clear = true, bool accumulate = true) {};
    /// accumulate $ df += (m M + b B + k K) dx $ (given the latest propagated displacement)
    void addMBKdx(core::MultiVecDerivId df, double m, double b, double k, bool clear = true, bool accumulate = true) {};
    /// accumulate $ df += (m M + b B + k K) velocity $
    void addMBKv(core::MultiVecDerivId df, double m, double b, double k, bool clear = true, bool accumulate = true) {};
    /// Add dt*Gravity to the velocity
    //void addSeparateGravity(double dt, core::MultiVecDerivId result( core::VecDerivId::velocity() ) );

    void computeContactForce(core::MultiVecDerivId result) {};
    void computeContactDf(core::MultiVecDerivId df) {};

    /*
            /// @}

            /// @name Matrix operations
            /// @{

            // BaseMatrix & BaseVector Computations
            virtual void getMatrixDimension(unsigned int * const, unsigned int * const, sofa::core::behavior::MultiMatrixAccessor* matrix = NULL);
            void getMatrixDimension(sofa::core::behavior::MultiMatrixAccessor* matrix)
            {
              getMatrixDimension(NULL, NULL, matrix);
            }
            void addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact);
            void multiVector2BaseVector(core::VecId src, defaulttype::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix);
            void multiVectorPeqBaseVector(core::VecId dest, defaulttype::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix);

            /// @}

            /// @name Debug operations
            /// @{

            /// Dump the content of the given vector.
            virtual void print( core::VecId v, std::ostream& out );
            virtual void printWithElapsedTime( core::VecId v,  unsigned time, std::ostream& out=std::cerr );
            */
};



}
}
}



#endif //SOFA_SIMULATION_COMMON_MECHANICALOPERATIONS_H
