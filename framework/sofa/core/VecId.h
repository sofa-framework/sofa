/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_VECID_H
#define SOFA_CORE_VECID_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>

#include <sstream>
#include <iostream>

namespace sofa
{

namespace core
{

/// Identify one vector stored in State
class VecId
{
public:
    enum { V_FIRST_DYNAMIC_INDEX = 9 }; ///< This is the first index used for dynamically allocated vectors
    enum Type
    {
        V_NULL=0,
        V_COORD,
        V_DERIV,
        V_CONST
    };
    Type type;
    unsigned int index;
    VecId(Type t, unsigned int i) : type(t), index(i) { }
    VecId() : type(V_NULL), index(0) { }
    bool isNull() const { return type==V_NULL; }
    static VecId null()          { return VecId(V_NULL, 0);}
    static VecId position()      { return VecId(V_COORD,0);}
    static VecId restPosition()  { return VecId(V_COORD,1);}
    static VecId velocity()      { return VecId(V_DERIV,0);}
    static VecId restVelocity()  { return VecId(V_DERIV,1);}
    static VecId force()         { return VecId(V_DERIV,3);}
    static VecId internalForce() { return VecId(V_DERIV,4);}
    static VecId externalForce() { return VecId(V_DERIV,5);}
    static VecId dx()            { return VecId(V_DERIV,6);}
    static VecId dforce()        { return VecId(V_DERIV,7);}
    static VecId accFromFrame()  { return VecId(V_DERIV,8);}
    static VecId freePosition()  { return VecId(V_COORD,2);}
    static VecId freeVelocity()  { return VecId(V_DERIV,2);}
    static VecId holonomicC()    { return VecId(V_CONST,0);}
    static VecId nonHolonomicC() { return VecId(V_CONST,1);}

    /// Test if two VecId identify the same vector
    bool operator==(const VecId& v) const
    {
        return type == v.type && index == v.index;
    }
    /// Test if two VecId identify the same vector
    bool operator!=(const VecId& v) const
    {
        return type != v.type || index != v.index;
    }

    std::string getName() const
    {
        std::string result;
        switch (type)
        {
        case VecId::V_NULL:
        {
            result+="NULL";
            break;
        }
        case VecId::V_COORD:
        {
            switch(index)
            {
            case 0: result+= "position";
                break;
            case 1: result+= "restPosition";
                break;
            case 2: result+= "freePosition";
                break;
                std::ostringstream out;
                out << index;
                result+= out.str();
                break;
            }
            result+= "(V_COORD)";
            break;
        }
        case VecId::V_DERIV:
        {
            switch(index)
            {
            case 0: result+= "velocity";
                break;
            case 1: result+= "restVelocity";
                break;
            case 2: result+= "freeVelocity";
                break;
            case 3: result+= "force";
                break;
            case 4: result+= "dx";
                break;
            case 5: result+= "accFromFrame";
                break;
            default:
                std::ostringstream out;
                out << index;
                result+= out.str();
                break;
            }
            result+= "(V_DERIV)";
            break;
        }
        case VecId::V_CONST:
        {
            switch(index)
            {
            case 0: result+= "holonomic";
                break;
            case 1: result+= "nonHolonolmic";
                break;
                std::ostringstream out;
                out << index;
                result+= out.str();
                break;
            }
            result+= "(V_CONST)";
            break;
        }
        }
        return result;
    }
};

inline std::ostream& operator << ( std::ostream& out, const VecId& v )
{
    out << v.getName();
    return out;
}

} // namespace core

} // namespace sofa

#endif
