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
#ifndef SOFA_COMPONENT_COLLISION_INCISEALONGPATHPERFORMER_H
#define SOFA_COMPONENT_COLLISION_INCISEALONGPATHPERFORMER_H

#include <sofa/component/collision/InteractionPerformer.h>

#include <sofa/component/collision/TopologicalChangeManager.h>
#include <sofa/component/collision/MouseInteractor.h>

namespace sofa
{

namespace component
{

namespace collision
{

class InciseAlongPathPerformerConfiguration
{
public:
    void setIncisionMethod (int m) {currentMethod=m;}
    void setSnapingBorderValue (int m) {snapingBorderValue = m;}
    void setSnapingValue (int m) {snapingValue = m;}

protected:
    int currentMethod;
    int snapingBorderValue;
    int snapingValue;

};


class SOFA_COMPONENT_COLLISION_API InciseAlongPathPerformer: public InteractionPerformer, public InciseAlongPathPerformerConfiguration
{
public:
    InciseAlongPathPerformer(BaseMouseInteractor *i):InteractionPerformer(i) {};

    void start();

    void execute();

    void draw() {};

protected:
    TopologicalChangeManager topologyChangeManager;
    BodyPicked startBody;
    BodyPicked firstBody;
    BodyPicked secondBody;
};
}
}
}

#endif
