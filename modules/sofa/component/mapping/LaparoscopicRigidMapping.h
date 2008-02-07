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
#ifndef SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/defaulttype/VecTypes.h>
namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class LaparoscopicRigidMapping : public BasicMapping, public virtual core::objectmodel::BaseObject
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    //typedef typename Coord::value_type Real;

public:
    Data<defaulttype::Vector3> pivot;
    Data<defaulttype::Quat> rotation;
    Data< component::topology::PointSubset > grab_index;

    LaparoscopicRigidMapping(In* from, Out* to)
        : Inherit(from, to)
        , pivot(initData(&pivot, defaulttype::Vector3(0,0,0), "pivot","TODO-pivot"))
        , rotation(initData(&rotation, defaulttype::Quat(0,0,0,1), "rotation", "TODO-rotation"))
        , grab_index(initData(&grab_index, "grab", "Index of the point to grab"))
        , mstate(NULL), grab_state(false)
    {
    }

    virtual ~LaparoscopicRigidMapping()
    {
        processRelease();
    }


    //void setPivot(const defaulttype::Vector3& val) { this->pivot = val; }
    //void setRotation(const defaulttype::Quat& val) { this->rotation = val; this->rotation.normalize(); }

    void init();

    virtual void reinit() {mstate = getMechanicalState();};

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void draw();
    void grab();

protected:
    void processGrab();
    void processRelease();
    //Contain the mechanical state of the tool
    core::componentmodel::behavior::MechanicalState< defaulttype::Vec3Types > *mstate;
    bool grab_state;/*
	  component::MechanicalObject<defaulttype::Vec3Types> *mm;*/

    //Find the DOFs of the laparascopic object
    core::componentmodel::behavior::MechanicalState< defaulttype::Vec3Types > *getMechanicalState()
    {
        sofa::simulation::tree::GNode * context = dynamic_cast< sofa::simulation::tree::GNode *>( this->getContext());
        if (context == NULL) return NULL;
        for (sofa::simulation::tree::GNode::ChildIterator it= context->child.begin(); it != context->child.end(); ++it)
        {
            if (core::componentmodel::behavior::MechanicalState< defaulttype::Vec3Types > *m = dynamic_cast< core::componentmodel::behavior::MechanicalState< defaulttype::Vec3Types > *>((*it)->getMechanicalState()))
                return m;

        }
        return NULL;
    }


    //For a give index, give the coordinates of a point
    helper::vector< defaulttype::Vec3f > getGrabPoints();
    sofa::helper::vector<core::componentmodel::behavior::BaseForceField*> forcefields;
    sofa::helper::vector<simulation::tree::GNode*> nodes;
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
