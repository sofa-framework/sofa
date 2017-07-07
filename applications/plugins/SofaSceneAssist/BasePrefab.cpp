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
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include <SofaSceneAssist/SceneAssist.h>
using sofa::SceneAssist ;

#include <SofaSceneAssist/BasePrefab.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

namespace _baseprefab_
{

BasePrefab::BasePrefab() :
    d_instancePath(initData(&d_instancePath, std::string("d"), "instancePath", "Path to the node containing the instance of this prefab"))
{}

BasePrefab::~BasePrefab() {}

void BasePrefab::init()
{
    Super::init() ;

    if( m_childNode.get() == nullptr) {
        BaseContext::SPtr context = getContext() ;
        if( context.get() )
        {
            m_childNode = SceneAssist::createNode(context, getName()+"Instance") ;
            d_instancePath.setValue(m_childNode->getPathName()) ;

            //SingleLink<Node, BasePrefab, BaseLink::FLAG_STOREPATH>* link =
            //new SingleLink<Node, BasePrefab, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK>(m_childNode->initLink("source", "The prefab that generates this node"), this) ;
        }

        /// We remove this component from the node and adds its to its instance holder.
        context->removeObject( this ) ;
        m_childNode->addObject( this ) ;
    }

    doInit(m_childNode) ;
}

void BasePrefab::reinit()
{
    Super::reinit() ;

    std::vector<BaseObject*> c;
    for(auto& aChild : m_childNode->getNodeObjects(c))
    {
        if(aChild != this)
            SceneAssist::deleteObjectFrom(m_childNode.get(), aChild);
    }

    for(auto& aChild : m_childNode->getChildren())
    {
        SceneAssist::deleteNode(m_childNode, aChild);
    }


    doReinit(m_childNode);
}

SOFA_DECL_CLASS(BasePrefab)

} // namespace _baseprefab_

} // namespace objectmodel

} // namespace core

} // namespace sofa

