#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONSTRAINT_H

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/helper/vector.h>


namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
class LagrangianMultiplierConstraint : public core::componentmodel::behavior::BaseConstraint
{
public:
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    class LMTypes
    {
    public:
        typedef typename DataTypes::Real Real;
        typedef Real Coord;
        typedef Real Deriv;
        typedef helper::vector<Real> VecReal;
        typedef helper::vector<Coord> VecCoord;
        typedef helper::vector<Deriv> VecDeriv;

        template <class T>
        class SparseData
        {
        public:
            SparseData(unsigned int _index, T& _data): index(_index), data(_data) {};
            unsigned int index;
            T data;
        };

        typedef SparseData<Coord> SparseCoord;
        typedef SparseData<Deriv> SparseDeriv;

        typedef helper::vector<SparseCoord> SparseVecCoord;
        typedef helper::vector<SparseDeriv> SparseVecDeriv;

        //! All the Constraints applied to a state Vector
        typedef	helper::vector<SparseVecDeriv> VecConst;


        template<typename real2>
        static void set(Coord& c, real2 x, real2 , real2 )
        {
            c = (Real)x;
        }

        template<typename real2>
        static void get(real2 &x, real2 &, real2 &, const Coord& c)
        {
            x = (Real)c;
        }

        template<typename real2>
        static void add(Coord& c, real2 x, real2 , real2 )
        {
            c += (Real)x;
        }

        static const char* Name()
        {
            return "LMTypes";
        }

        static unsigned int size() {return 0;};

        Real& operator()(int i)
        {
            Real r(0.0);
            return r;
        }

        /// Const access to i-th element.
        const Real& operator()(int i) const
        {
            Real r(0.0);
            return r;
        }

    };
    typedef typename LMTypes::VecCoord LMCoord;
    typedef typename LMTypes::VecDeriv LMDeriv;

protected:
    /// Langrange multipliers
    component::MechanicalObject<LMTypes>* lambda;

public:

    LagrangianMultiplierConstraint()
    {
        lambda = new component::MechanicalObject<LMTypes>;
    }

    ~LagrangianMultiplierConstraint()
    {
        delete lambda;
    }

    virtual void projectResponse() ///< project dx to constrained space
    {
    }
    virtual void projectVelocity() {} ///< project v to constrained space (v models a velocity)
    virtual void projectPosition() {} ///< project x to constrained space (x models a position)
    virtual void projectFreeVelocity() {} ///< project vFree to constrained space (vFree models a velocity)
    virtual void projectFreePosition() {} ///< project xFree to constrained space (xFree models a position)

    virtual core::componentmodel::behavior::BaseMechanicalState* getDOFs()
    {
        return lambda;
    }
};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
