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
#include <sofa/core/behavior/MechanicalMatrix.h>
#include <iostream>
namespace sofa::core::behavior
{

std::ostream& operator << (std::ostream& out, const MechanicalMatrix& m )
{
    out << '(';
    bool first = true;
    for (unsigned int i=0; i<m.factors.size(); ++i)
    {
        const SReal f = m.factors[i];
        if (f!=0.0)
        {
            if (!first) out << ' ';
            if (f == -1.0) out << '-';
            else if (f < 0) out << f << ' ';
            else
            {
                if (!first) out << '+';
                if (f != 1.0) out << f << ' ';
            }
            out << ("MBK")[i];
            first = false;
        }
    }
    out << ')';
    return out;
}

const MechanicalMatrix MechanicalMatrix::M(1,0,0);
const MechanicalMatrix MechanicalMatrix::B(0,1,0);
const MechanicalMatrix MechanicalMatrix::K(0,0,1);

} /// namespace sofa::core::behavior

