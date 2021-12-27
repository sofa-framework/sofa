/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/core/config.h>
#include <sofa/type/Vec.h>
#include <iosfwd>

namespace sofa::core::behavior
{

namespace
{
    using sofa::type::Vec3;
}

/// Helper class allowing to construct mechanical expressions
///
class SOFA_CORE_API MechanicalMatrix
{
protected:
    enum { MFACT = 0, BFACT = 1, KFACT = 2 };
    Vec3 factors;
public:
    MechanicalMatrix(SReal m, SReal b, SReal k) : factors(m,b,k) {}
    explicit MechanicalMatrix(const Vec3& f) : factors(f) {}

    static const MechanicalMatrix M;
    static const MechanicalMatrix B;
    static const MechanicalMatrix K;

    SReal getMFact() const { return factors[MFACT]; }
    SReal getBFact() const { return factors[BFACT]; }
    SReal getKFact() const { return factors[KFACT]; }

    MechanicalMatrix operator + (const MechanicalMatrix& m2) const { return MechanicalMatrix(factors + m2.factors); }
    MechanicalMatrix operator - (const MechanicalMatrix& m2) const { return MechanicalMatrix(factors - m2.factors); }
    MechanicalMatrix operator - () const { return MechanicalMatrix(- factors); }
    MechanicalMatrix operator * (SReal f) const { return MechanicalMatrix(factors * f); }
    MechanicalMatrix operator / (SReal f) const { return MechanicalMatrix(factors / f); }
    friend SOFA_CORE_API std::ostream& operator << (std::ostream& out, const MechanicalMatrix& m );
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const MechanicalMatrix& m );

} /// namespace sofa::core::behavior

