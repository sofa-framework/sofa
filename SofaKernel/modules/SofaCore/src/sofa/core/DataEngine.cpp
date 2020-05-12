/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/core/DataEngine.h>

namespace sofa
{

namespace core
{

DataEngine::DataEngine()
{
}

DataEngine::~DataEngine()
{
}


/// Add a new input to this engine
void DataEngine::addInput(objectmodel::BaseData* n)
{
    m_dataTracker.trackData(*n);
    //TODO(dmarchal:2020-05-11) This is hacky as hell. Being part of a group is something
    // the is part of the data, not something you can change. If you need
    // to display the input/outputs of a DataEngine node implement a custom widgets
    // that iterate over the input/outputs instead of trying to fake that by regrouping that.
    if (n->getOwner() == this && n->getGroup().empty())
        n->setGroup("Inputs"); // set the group of input Datas if not yet set
    core::objectmodel::DDGNode::addInput(n);
    setDirtyValue();
}

/// Add a new output to this engine
void DataEngine::addOutput(objectmodel::BaseData* n)
{
    //TODO(dmarchal:2020-05-11) This is hacky as hell. Being part of a group is something
    // the is part of the data, not something you can change. If you need
    // to display the input/outputs of a DataEngine node implement a custom widgets
    // that iterate over the input/outputs instead of trying to fake that by regrouping that.
    if (n->getOwner() == this && n->getGroup().empty())
        n->setGroup("Outputs"); // set the group of output Datas if not yet set
    core::objectmodel::DDGNode::addOutput(n);
    setDirtyValue();
}

void DataEngine::delInput(sofa::core::objectmodel::BaseData *data)
{
    core::objectmodel::DDGNode::doDelInput(data);
}

void DataEngine::delOutput(sofa::core::objectmodel::BaseData *data)
{
    core::objectmodel::DDGNode::doDelOutput(data);
}

void DataEngine::updateAllInputs()
{
    for(auto& input : getInputs())
    {
        input->updateIfDirty();
    }
}

void DataEngine::update()
{
    updateAllInputs();
    DDGNode::cleanDirty();
    doUpdate();
    m_dataTracker.clean();
}



} // namespace core

} // namespace sofa
