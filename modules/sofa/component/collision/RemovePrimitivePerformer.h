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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_REMOVEPRIMITIVEPERFORMER_H
#define SOFA_COMPONENT_COLLISION_REMOVEPRIMITIVEPERFORMER_H


#include <sofa/component/collision/InteractionPerformer.h>
#include <sofa/component/collision/TopologicalChangeManager.h>
#include <sofa/component/collision/MouseInteractor.h>
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

template <class DataTypes>
class SOFA_COMPONENT_COLLISION_API RemovePrimitivePerformer: public       TInteractionPerformer<DataTypes>, public RemovePrimitivePerformerConfiguration
{
    typedef typename DataTypes::Coord              Coord;
    typedef typename DataTypes::VecCoord           VecCoord;
    typedef sofa::helper::vector <unsigned int>    VecIds;


public:
    RemovePrimitivePerformer(BaseMouseInteractor *i);
    ~RemovePrimitivePerformer() {}


    void start();
    void execute();
    void end();
    void draw();

protected:

    void createElementList ();

    VecIds getNeighboorElements (VecIds& elementsToTest);

    VecIds getElementInZone (VecIds& elementsToTest);


protected:
    sofa::component::collision::TopologicalChangeManager topologyChangeManager;

    core::componentmodel::behavior::MechanicalState<DataTypes>* mstateCollision;
    BodyPicked picked;
    bool firstClick;

    VecIds selectedElem;
    //VecIds rejectedElem;
    //VecIds testElem;
    //VecIds tmp_testElem;//?

    sofa::core::componentmodel::topology::TopologyObjectType topoType;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_COLLISION_REMOVEPRIMITIVEPERFORMER_CPP)
extern template class SOFA_COMPONENT_COLLISION_API RemovePrimitivePerformer<defaulttype::Vec3Types>;
#endif

}
}
}

#endif
