/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_H
#define SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/io/bvh/BVHLoader.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace container
{

class ArticulationCenter;
class Articulation;

/**
* This class allow to store and retrieve all the articulation centers from an articulated rigid object
* @see ArticulatedCenter
* @see Articulation
*/
class SOFA_GENERAL_RIGID_API ArticulatedHierarchyContainer : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(ArticulatedHierarchyContainer,core::objectmodel::BaseObject);

    typedef defaulttype::SolidTypes<SReal>::Transform Transform;

protected:
    ArticulatedHierarchyContainer();

    ~ArticulatedHierarchyContainer() {}
public:


    void init();

    void setFilename(std::string f) {filename.setValue(f);}

    helper::vector<ArticulationCenter*> getArticulationCenters() { return articulationCenters; }
    ArticulationCenter* getArticulationCenterAsChild(int index);
    helper::vector<ArticulationCenter*> getAcendantList(int index);

    helper::vector<ArticulationCenter*> articulationCenters;
    helper::vector<ArticulationCenter*> acendantList;

    bool chargedFromFile;
    int numOfFrames;
    double dtbvh;

protected:
    sofa::core::objectmodel::DataFileName filename;
private:


    unsigned int id;
    sofa::helper::io::bvh::BVHJoint* joint;
    void buildCenterArticulationsTree(sofa::helper::io::bvh::BVHJoint*, int id_buf, const char* name, simulation::Node* node);
};

/**
*	This class defines an articulation center.	This contains a set of articulations.
*	An articulation center is always defined between two DOF's (ParentDOF and ChildDOF).
*	It stores the local position of the center articulation in relation to this DOF's (posOnParent, posOnChild),
*	theirs indices (parentIndex, childIndex) and the global position of the articulation center.
*	The local positions and indices have to be provided at initialization.
*	For the same articulation center can be defined several articulations.
*	All the variables which are defined in this class can be modified once sofa is running.
*/

class SOFA_GENERAL_RIGID_API ArticulationCenter : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(ArticulationCenter,core::objectmodel::BaseObject);

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
protected:
    ArticulationCenter();
    ~ArticulationCenter() {}
public:
    /**
    *	All DOF's can be identified, in an univocal way, by an index
    *	this variable will store the index of the parentDOF of the articulation center
    */
    Data<int> parentIndex;
    /**
    *	All DOF's can be identified, in an univocal way, by an index
    *	this variable will store the index of the childDOF of the articulation center
    */
    Data<int> childIndex;
    /**
    *	Global position for the articulation center. It's not necessary to provide it at initialization.
    *	This will be computed in mapping using the global position of the parent DOF and the local position
    *	of the center articulation
    */
    Data<defaulttype::Vector3> globalPosition;
    /**
    *	It stores the local position of the center articulation in relation to the global position of the parentDOF
    */
    Data<defaulttype::Vector3> posOnParent;
    /**
    *	It stores the local position of the center articulation in relation to the global position of the childDOF
    */
    Data<defaulttype::Vector3> posOnChild;
    /**
    *	It tells if the articulations of the articulation center are processed one by one or globally:
    *   0 - (default         ) articulation are treated one by one, the axis of the second articulation is updated by the potential rotation of the first articulation
    						   potential problems could arise when rotation exceed 90° (known problem of euler angles)
    *   1 - (Attach on Parent) the axis of the articulations are linked to the parent - rotations are treated by successive increases -
    *   2 - (Attach on Child ) the axis of the articulations are linked to the child (estimate position from the previous time step) - rotations are treated by successive increases -
    */
    Data<int> articulationProcess;

    /**
    *   for ARBORIS Mapping
    *	Store information about the transformation induced by the articulation center (joint)
    *   H_p_pLc and H_c_cLp redefine posOnParent, posOnChild (a local rotation between the center of the articualtion and the parent/child bodies can be defined)
    *   H_pLc_cLp is transformation induced by the articulations of the articulation center (joint)
    			  it is updated during "apply" function of the Mapping
    */
    ArticulatedHierarchyContainer::Transform H_p_pLc, H_c_cLp, H_pLc_cLp;


    helper::vector<Articulation*> articulations;

    defaulttype::Vector3 posOnChildGlobal(defaulttype::Quat localToGlobal)
    {
        defaulttype::Vector3 result = localToGlobal.rotate(posOnChild.getValue());
        return result;
    }

    defaulttype::Vector3 initTranslateChild(defaulttype::Quat objectRotation)
    {
        defaulttype::Vector3 PAParent = posOnParent.getValue() - defaulttype::Vector3(0,0,0);
        defaulttype::Vector3 PAChild = posOnChild.getValue() - defaulttype::Vector3(0,0,0);
        return objectRotation.rotate(PAParent - PAChild);
    }

    defaulttype::Vector3 translateChild(defaulttype::Quat object1Rotation, defaulttype::Quat object2Rotation)
    {
        defaulttype::Vector3 APChild = defaulttype::Vector3(0,0,0) - posOnChild.getValue();
        defaulttype::Vector3 AP1 = object2Rotation.rotate(APChild);
        defaulttype::Vector3 AP2 = object1Rotation.rotate(AP1);
        return AP2 - AP1;
    }

    defaulttype::Vector3 correctPosChild(defaulttype::Vector3 object1Pos, defaulttype::Quat object1Rot, defaulttype::Vector3 object2Pos, defaulttype::Quat object2Rot)
    {
        defaulttype::Vector3 result;
        defaulttype::Vector3 PAParent = posOnParent.getValue() - defaulttype::Vector3(0,0,0);
        defaulttype::Vector3 PAChild = posOnChild.getValue() - defaulttype::Vector3(0,0,0);
        defaulttype::Vector3 A1 = object1Pos + object1Rot.rotate(PAParent);
        defaulttype::Vector3 A2 = object2Pos + object2Rot.rotate(PAChild);

        result = A1 - A2;

        return result;

    }

    helper::vector<Articulation*>& getArticulations() { return articulations; }

    defaulttype::Quat OrientationArticulationCenter;
    defaulttype::Vector3 DisplacementArticulationCenter;
    defaulttype::Vector3 Disp_Rotation;


}; // end ArticulationCenter

/**
*	This class defines an articulation.
*	An articulation is defined by an axis, an orientation and an index.
*	All the variables which are defined in this class can be modified once sofa is running.
*/
class SOFA_GENERAL_RIGID_API Articulation : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(Articulation,core::objectmodel::BaseObject);

    /**
    *	An articulation is defined by an axis, an orientation and an index.
    *	@param axis is a Vector3. It determines the motion axis
    *	@param rotation is a boolean. If true, it defines a rotation motion. Otherwise it does nothing.
    *	@param translation is a boolean. If true, it defines a translation motion. Otherwise it does nothing.
    *	@param articulationIndex is an integer. This index identifies, in an univocal way, one articulation
    *	from the set of articulations of a rigid object.
    */
protected:
    Articulation();
    ~Articulation() {}
public:
    /**
    *	this variable defines the motion axis
    */
    Data<defaulttype::Vector3> axis;
    /**
    *	If true, this variable sets a rotation motion
    *	otherwise it does nothing
    */
    Data<bool> rotation;
    /**
    *	If true, this variable sets a translation motion
    *	otherwise it does nothing
    */
    Data<bool> translation;
    /**
    *	This is global index to number the articulations
    */
    Data<int> articulationIndex;

    std::vector<double> motion;

    /**
     *	For Arboris Mapping H_pLc_a : transformation accumulates the successive transformation provided by articulations on the
     *  same articulation center
     */
    ArticulatedHierarchyContainer::Transform H_pLc_a;
};

} // namespace container

} // namespace component

} // namespace sofa

#endif

