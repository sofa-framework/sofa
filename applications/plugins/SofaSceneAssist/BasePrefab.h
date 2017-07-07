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

#ifndef SOFASCENEASSIST_BASEPREFAB_H
#define SOFASCENEASSIST_BASEPREFAB_H

#include <SofaSimulationGraph/DAGNode.h>
#include <sofa/core/BehaviorModel.h>
#include <SofaSceneAssist/config.h>



namespace sofa
{

namespace core
{

namespace objectmodel
{

namespace _baseprefab_
{

using sofa::core::objectmodel::BaseObject ;
using sofa::simulation::graph::DAGNode ;
using sofa::simulation::Node ;

class BasePrefab : public BaseObject
{
public:
    SOFA_CLASS(BasePrefab, BaseObject);

    typedef BaseObject Super;

    /////////////////// Inherited from BaseObject ////////////////////
    virtual void init() override ;
    virtual void reinit() override ;
    //////////////////////////////////////////////////////////////////

    virtual void doInit(Node::SPtr& prefabInstance) = 0 ;
    virtual void doReinit(Node::SPtr& prefabInstance) = 0 ;

protected:
    BasePrefab() ;
    virtual ~BasePrefab() ;

    Node::SPtr m_childNode ;
    Data<std::string> d_instancePath ;
};

} // namespace _baseprefab_

using _baseprefab_::BasePrefab ;

} // namespace objectmodel

} // namespace core

} // namespace sofa


#endif // SOFASCENEASSIST_BASEPREFAB_H
