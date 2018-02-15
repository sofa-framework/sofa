/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_QTOGREVIEWER_PROPAGATEOGRESCENEMANAGER_H
#define SOFA_QTOGREVIEWER_PROPAGATEOGRESCENEMANAGER_H

#include <sofa/simulation/Visitor.h>
#include <Ogre.h>


namespace sofa
{
namespace core
{
namespace objectmodel
{
class BaseObject;
}
}
namespace simulation
{
class Node;
namespace ogre
{
class PropagateOgreSceneManager : public Visitor
{

public:
    PropagateOgreSceneManager(const core::ExecParams* params,Ogre::SceneManager* sceneMgr);
    Visitor::Result processNodeTopDown(simulation::Node* );


    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const { return "ogre"; }
    const char* getClassName() const { return "OgreInitVisitor"; }
protected:
    void processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* );

    Ogre::SceneManager*  mSceneMgr;
};
} // ogre
} // simulation
} // sofa
#endif //SOFA_QTOGREVIEWER_PROPAGATEOGRESCENEMANAGER_H
