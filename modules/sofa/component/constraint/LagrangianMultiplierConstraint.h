#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONSTRAINT_H

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/component/MechanicalObject.h>
#include <vector>


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
        typedef ::vector<Real> VecReal;
        typedef ::vector<Coord> VecCoord;
        typedef ::vector<Deriv> VecDeriv;

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

        typedef ::vector<SparseCoord> SparseVecCoord;
        typedef ::vector<SparseDeriv> SparseVecDeriv;

        //! All the Constraints applied to a state Vector
        typedef	::vector<SparseVecDeriv> VecConst;


        static void set(Coord& c, double x, double , double )
        {
            c = (Real)x;
        }

        static void get(double &x, double &, double &, const Coord& c)
        {
            x = (double)c;
        }

        static void add(Coord& c, double x, double , double )
        {
            c += (Real)x;
        }

        static const char* Name()
        {
            return "LMTypes";
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
    virtual void projectVelocity() {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition() {} ///< project x to constrained space (x models a position)

    virtual core::componentmodel::behavior::BaseMechanicalState* getDOFs()
    {
        return lambda;
    }
};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
