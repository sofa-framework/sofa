/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERCONSTRAINT_H

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/behavior/Constraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/helper/vector.h>


namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
class LagrangianMultiplierConstraint : public core::behavior::BaseConstraint
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LagrangianMultiplierConstraint,DataTypes),core::behavior::BaseConstraint);

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

        static Coord interpolate(const helper::vector< Coord > &ancestors, const helper::vector< Real > &coefs)
        {
            assert(ancestors.size() == coefs.size());

            Coord c = (Real)0.0;

            for (unsigned int i = 0; i < ancestors.size(); i++)
            {
                c += ancestors[i] * coefs[i];
            }

            return c;
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

    virtual core::behavior::BaseMechanicalState* getDOFs()
    {
        return lambda;
    }

    /// this constraint is holonomic
    bool isHolonomic() {return true;}

};

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
