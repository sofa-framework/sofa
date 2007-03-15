/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMECHANICALSTATE_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMECHANICALSTATE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <iostream>

using namespace sofa::helper;

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class BaseMechanicalMapping;

class BaseMechanicalState : public virtual objectmodel::BaseObject
{
public:
    BaseMechanicalState ()
        : objectmodel::BaseObject()
    {}
    virtual ~BaseMechanicalState ()
    { }

    virtual void resize(int vsize) = 0;

    virtual void init() = 0;

    /// @name Integration related methods
    /// @{

    virtual void beginIntegration(double /*dt*/) { }

    virtual void endIntegration(double /*dt*/) { }

    virtual void resetForce() =0;//{ vOp( VecId::force() ); }

    virtual void resetConstraint() =0;

    virtual void accumulateForce() { }

    virtual void accumulateDf() { }


    class VecId
    {
    public:
        enum { V_FIRST_DYNAMIC_INDEX = 4 }; ///< This is the first index used for dynamically allocated vectors
        enum Type
        {
            V_NULL=0,
            V_COORD,
            V_DERIV
        };
        Type type;
        unsigned int index;
        VecId(Type t, unsigned int i) : type(t), index(i) { }
        VecId() : type(V_NULL), index(0) { }
        bool isNull() const { return type==V_NULL; }
        static VecId null()     { return VecId(V_NULL,0); }
        static VecId position() { return VecId(V_COORD,0); }
        static VecId initialPosition() { return VecId(V_COORD,1); }
        static VecId velocity() { return VecId(V_DERIV,0); }
        static VecId initialVelocity() { return VecId(V_DERIV,3); }
        static VecId force() { return VecId(V_DERIV,1); }
        static VecId dx() { return VecId(V_DERIV,2); }
        bool operator==(const VecId& v)
        {
            return type == v.type && index == v.index;
        }
    };

    virtual void vAlloc(VecId v) = 0; // {}

    virtual void vFree(VecId v) = 0; // {}

    virtual void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), double f=1.0) = 0; // {}

    virtual double vDot(VecId a, VecId b) = 0; //{ return 0; }

    virtual void setX(VecId v) = 0; //{}

    virtual void setV(VecId v) = 0; //{}

    virtual void setF(VecId v) = 0; //{}

    virtual void setDx(VecId v) = 0; //{}

    virtual void setC(VecId v) = 0; //{}

    // new : get compliance on the constraints
    virtual void getCompliance(double /*dt*/, double ** /*w*/, double * /*dfree*/, int& /*numContact*/) { }
    // apply contact force AND compute the subsequent dX
    virtual void applyContactForce(double * /*f*/) { }

    /// @}

    /// @name Data output
    /// @{
    virtual void printDOF( VecId, std::ostream& =std::cerr ) = 0;
    virtual void initGnuplot() {}
    virtual void exportGnuplot(double) {}
    virtual unsigned printDOFWithElapsedTime(VecId, unsigned =0, unsigned =0, std::ostream& =std::cerr ) {return 0;};
    /// @}


    /*! \fn void addBBox()
     *  \brief Used to add the bounding-box of this mechanical model to the given bounding box.
     *
     *  Note that if it does not make sense for this particular object (such as if the DOFs are not 3D), then the default implementation displays a warning message and returns false.
     */
    //virtual bool addBBox(double* /*minBBox*/, double* /*maxBBox*/)
    //{
    //  std::cerr << "warning: unumplemented method MechanicalState::addBBox() called.\n";
    //  return false;
    //}
};

inline std::ostream& operator<<(std::ostream& o, const BaseMechanicalState::VecId& v)
{
    switch (v.type)
    {
    case BaseMechanicalState::VecId::V_NULL: o << "vNull"; break;
    case BaseMechanicalState::VecId::V_COORD: o << "vCoord"; break;
    case BaseMechanicalState::VecId::V_DERIV: o << "vDeriv"; break;
    default: o << "vUNKNOWN"; break;
    }
    o << '[' << v.index << ']';
    return o;
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
