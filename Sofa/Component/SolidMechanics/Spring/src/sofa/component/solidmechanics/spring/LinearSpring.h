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
#include <ostream>
#include <istream>
#include <sofa/component/solidmechanics/spring/config.h>

namespace sofa::component::solidmechanics::spring
{

/// This class contains the description of one linear spring
template<class T>
class LinearSpring
{
public:
    using Real = T;

    sofa::Index m1, m2;     ///< the two extremities of the spring: masses m1 and m2
    Real ks;                ///< spring stiffness
    Real kd;                ///< damping factor
    Real initpos;           ///< rest length of the spring
    bool elongationOnly;    ///< only forbid elongation, not compression
    bool enabled;           ///< false to disable this spring (i.e. broken)

    explicit LinearSpring(const sofa::Index m1 = 0, const sofa::Index m2 = 0,
                          Real ks = 0.0, Real kd = 0.0, Real initpos = 0.0,
                          const bool noCompression = false,
                          const bool enabled = true)
        : m1(m1), m2(m2), ks(ks), kd(kd), initpos(initpos),
          elongationOnly(noCompression), enabled(enabled)
    {}

    friend std::istream& operator >> ( std::istream& in, LinearSpring<Real>& s )
    {
        in >> s.m1 >> s.m2 >> s.ks >> s.kd >> s.initpos;
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const LinearSpring<Real>& s )
    {
        out << s.m1 << " " << s.m2 << " " << s.ks << " " << s.kd << " " << s.initpos;
        return out;
    }
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_LINEARSPRING_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API LinearSpring<SReal>;
#endif

}
