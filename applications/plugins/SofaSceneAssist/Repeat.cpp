/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/******************************************************************************
*  Contributors:                                                              *
*  - damien.marchal@univ-lille1.fr                                            *
******************************************************************************/
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseContext ;

#include <sofa/core/objectmodel/BaseNode.h>
using sofa::core::objectmodel::BaseNode ;

#include <sofa/core/objectmodel/BaseObjectDescription.h>
using sofa::core::objectmodel::BaseObjectDescription ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::RegisterObject ;

#include <SofaSceneAssist/BasePrefab.h>
using sofa::core::objectmodel::BasePrefab ;

#include <SofaSceneAssist/config.h>

#include <SofaSceneAssist/SceneAssist.h>
using sofa::SceneAssist ;

namespace sofa
{

namespace component
{

namespace _repeat_
{

class Repeat : public BasePrefab
{

public:
    SOFA_CLASS(Repeat, BasePrefab);

    virtual void doInit(Node::SPtr& prefabInstance) override ;
    virtual void doReinit(Node::SPtr& prefabInstance) override ;

protected:
    Repeat() ;
    virtual ~Repeat() ;

private:
    Data<int>  d_numChild ;
    Data<int>  d_numObject ;
};

Repeat::Repeat() :
     d_numChild ( initData(&d_numChild, 0, "numChild", "Number of child node"))
    ,d_numObject ( initData(&d_numObject, 0, "numObject", "Number of objects in nodes"))
{
    d_numChild.setGroup("Prefab") ;
    d_numObject.setGroup("Prefab") ;
}

Repeat::~Repeat(){}

void Repeat::doInit(Node::SPtr &prefabInstance)
{
    doReinit(prefabInstance) ;
}

void Repeat::doReinit(Node::SPtr &prefabInstance)
{
    for(int i=0;i<d_numChild.getValue();i++)
    {
        std::stringstream tmp;
        tmp << "child_" << i ;
        auto childNode = SceneAssist::createNode(prefabInstance,  tmp.str() ) ;
        for(unsigned int j=0;j<d_numObject.getValue();j++)
            SceneAssist::createObject(childNode, "MechanicalObject", {{"name", "obj"}}) ;
    }
}

SOFA_DECL_CLASS(Repeat)
int RepeatClass = core::RegisterObject("Repeat.")
        .add< Repeat >();

} // namespace _baseprefab_

} // namespace component

} // namespace sofa
