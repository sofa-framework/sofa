/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_COLLISION_REMOVEPRIMITIVEPERFORMER_H
#define SOFA_COMPONENT_COLLISION_REMOVEPRIMITIVEPERFORMER_H
#include "config.h"

#include <SofaUserInteraction/InteractionPerformer.h>
#include <SofaUserInteraction/TopologicalChangeManager.h>
#include <SofaUserInteraction/MouseInteractor.h>
/*
#include <sofa/core/CollisionModel.h>
#include <sofa/core/CollisionElement.h>
*/
namespace sofa
{

namespace component
{

namespace collision
{

/** Class to configure primitive removal. Several parameters:
  * - topologicalOperation: if 0, other parameters arn't use.
  *   0 = "remove on element"
  *   1 = "remove a zone of elements"
  * - volumicMesh:
  *   false = surfacique mesh
  *   true = volumique mesh
  * - selectorScale: size of zone
  */
class RemovePrimitivePerformerConfiguration
{
public:
    RemovePrimitivePerformerConfiguration():topologicalOperation(0) {}
    void setTopologicalOperation (int m) {topologicalOperation = m;}
    void setVolumicMesh (bool v) {volumicMesh = v;}
    void setScale (double s) {selectorScale = s;}

protected:
    int topologicalOperation;
    bool volumicMesh;
    double selectorScale;

};


/** Class to perform removing of topological elements (either one element or a an area) and handling topological mapping
  */
template <class DataTypes>
class SOFA_USER_INTERACTION_API RemovePrimitivePerformer: public       TInteractionPerformer<DataTypes>, public RemovePrimitivePerformerConfiguration
{
    typedef typename DataTypes::Real               Real;
    typedef typename DataTypes::Coord              Coord;
    typedef typename DataTypes::VecCoord           VecCoord;
    typedef sofa::helper::vector <unsigned int>    VecIds;

public:
    RemovePrimitivePerformer(BaseMouseInteractor *i);
    ~RemovePrimitivePerformer() {}

    /// Functions called by TopologicalOperation performer
    void start();
    void execute();
    void end();
    void draw(const core::visual::VisualParams* vparams);

protected:

    /** Function creating a list of elements concerned by the removal operation.
      * This function detect if a volume or a surface or a volume on the surface is going to be removed.
      * Elements are stored in @see selectedElem.
      * Calling @see getNeighboorElements and @see getElementInZone.
      *
      * @return bool: false if method has encounter an error.
      */
    bool createElementList ();


    /** Function to get all elements directly neighboor of a given list of elements
      * compute the list without redundancy using container xxAroundVertex() (where xx is the type of element).
      * @param elementsToTest: vector of element Id to test.
      * @return VecIds: vector of element Id containing neighbour (without already accepted elements and redundancy).
      */
    VecIds getNeighboorElements (VecIds& elementsToTest);


    /** Function testing if elements are in the range of a given zone
      * The zone is given by the selectorScale.
      * Test is done on Barycentric point of elements. I.e if this point is in the range of the area. then, element is accepted otherwise, element is rejected.
      * @param elementsToTest: vector of element Id to test.
      * @return VecIds: vector of element Id containing accepted element.
      */
    VecIds getElementInZone (VecIds& elementsToTest);



protected:
    /// picked structure from mouseInteractor.
    BodyPicked picked;

    /// bool: true if first click (when removing zone, first clic show zone, second delete it).
    bool firstClick;

    /// bool: true if a surface zone is going to be removed on a volumique mesh.
    bool surfaceOnVolume;

    /// bool: true if a volumique zone is going to be removed at the surface of a volumique mesh.
    bool volumeOnSurface;

    /// vector of element Id concerned by the operation
    VecIds selectedElem;

private:
    /// Class containing removal functions (given collision model)
    sofa::component::collision::TopologicalChangeManager topologyChangeManager;
    /// Point to collision class
    core::behavior::MechanicalState<DataTypes>* mstateCollision;
    /// Enum storing the type to current topolgy: TRIANGLE, QUAD, TETRAHEDRON or HEXAHEDRON
    sofa::core::topology::TopologyObjectType topoType;
    /// Pointer to current topology detect by picking
    sofa::core::topology::BaseMeshTopology* topo_curr;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_REMOVEPRIMITIVEPERFORMER_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_USER_INTERACTION_API RemovePrimitivePerformer<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_USER_INTERACTION_API RemovePrimitivePerformer<defaulttype::Vec3dTypes>;
#endif
#endif

}
}
}

#endif
