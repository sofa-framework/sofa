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
#ifndef SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_H
#define SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;
using namespace sofa::simulation::tree;

/**
* This class allow to store and retrieve all the articulation centers from an articulated rigid object
* @see ArticulatedCenter
* @see Articulation
*/
class ArticulatedHierarchyContainer : public virtual core::objectmodel::BaseObject
{
public:

    /**
    *	This class defines an articulation center.	This contains a set of articulations.
    *	An articulation center is always defined between two DOF's (ParentDOF and ChildDOF).
    *	It stores the local position of the center articulation in relation to this DOF's (posOnParent, posOnChild),
    *	theirs indices (parentIndex, childIndex) and the global position of the articulation center.
    *	The local positions and indices have to be provided at initialization.
    *	For the same articulation center can be defined several articulations.
    *	All the variables which are defined in this class can be modified once sofa is running.
    */

    class ArticulationCenter : public virtual core::objectmodel::BaseObject
    {
    public:

        /**
        *	This class defines an articulation.
        *	An articulation is defined by an axis, an orientation and an index.
        *	All the variables which are defined in this class can be modified once sofa is running.
        */
        class Articulation : public virtual core::objectmodel::BaseObject
        {
        public:

            /**
            *	An articulation is defined by an axis, an orientation and an index.
            *	@param axis is a Vector3. It determines the motion axis
            *	@param rotation is a boolean. If true, it defines a rotation motion. Otherwise it does nothing.
            *	@param translation is a boolean. If true, it defines a translation motion. Otherwise it does nothing.
            *	@param articulationIndex is an integer. This index identifies, in an univocal way, one articulation
            *	from the set of articulations of a rigid object.
            */
            Articulation();
            ~Articulation() {};

            /**
            *	this variable defines the motion axis
            */
            DataField<Vector3> axis;
            /**
            *	If true, this variable sets a rotation motion
            *	otherwise it does nothing
            */
            DataField<bool> rotation;
            /**
            *	If true, this variable sets a translation motion
            *	otherwise it does nothing
            */
            DataField<bool> translation;
            /**
            *	This is global index to number the articulations
            */
            DataField<int> articulationIndex;
        };

        /**
        *	This class contain a set of articulations.
        *	@see Articulation
        *	@param parentIndex. It stores the index of the parentDOF of the articulation center
        *	@param childIndex. It stores the index of the childDOF of the articulation center
        *	@param posOnParent. It stores the local position of the center articulation in relation
        *	to the global position of the parentDOF
        *	@param posOnChild. It stores the local position of the center articulation in relation
        *	to the global position of the childDOF
        */
        ArticulationCenter();
        ~ArticulationCenter() {};

        /**
        *	All DOF's can be identified, in an univocal way, by an index
        *	this variable will store the index of the parentDOF of the articulation center
        */
        DataField<int> parentIndex;
        /**
        *	All DOF's can be identified, in an univocal way, by an index
        *	this variable will store the index of the childDOF of the articulation center
        */
        DataField<int> childIndex;
        /**
        *	Global position for the articulation center. It's not necessary to provide it at initialization.
        *	This will be computed in mapping using the global position of the parent DOF and the local position
        *	of the center articulation
        */
        DataField<Vector3> globalPosition;
        /**
        *	It stores the local position of the center articulation in relation to the global position of the parentDOF
        */
        DataField<Vector3> posOnParent;
        /**
        *	It stores the local position of the center articulation in relation to the global position of the childDOF
        */
        DataField<Vector3> posOnChild;

        Vector3 initTranslateChild(Quat objectRotation)
        {
            Vector3 PAParent = posOnParent.getValue() - Vector3(0,0,0);
            Vector3 PAChild = posOnChild.getValue() - Vector3(0,0,0);
            return objectRotation.rotate(PAParent - PAChild);
        }

        Vector3 translateChild(Quat object1Rotation, Quat object2Rotation)
        {
            Vector3 APChild = Vector3(0,0,0) - posOnChild.getValue();
            Vector3 AP1 = object2Rotation.rotate(APChild);
            Vector3 AP2 = object1Rotation.rotate(AP1);
            return AP2 - AP1;
        }

        const vector<Articulation*> getArticulations();
    };

    ArticulatedHierarchyContainer() {};
    ~ArticulatedHierarchyContainer() {};

    const vector<ArticulationCenter*> getArticulationCenters();
};

} // namespace container

} // namespace component

} // namespace sofa

#endif

