/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/core/DataTracker.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>

namespace sofa::core
{


void DataTracker::trackData( const objectmodel::BaseData& data )
{
    m_dataTrackers[&data] = data.getCounter();
}

bool DataTracker::hasChanged( const objectmodel::BaseData& data ) const
{
    if (m_dataTrackers.find(&data) != m_dataTrackers.end())
        return m_dataTrackers.at(&data) != data.getCounter();
    return false;
}

bool DataTracker::hasChanged() const
{
    for( DataTrackers::const_iterator it=m_dataTrackers.begin(),itend=m_dataTrackers.end() ; it!=itend ; ++it )
        if( it->second != it->first->getCounter() ) return true;
    return false;
}

void DataTracker::clean( const objectmodel::BaseData& data )
{
    m_dataTrackers[&data] = data.getCounter();
}

void DataTracker::clean()
{
    for( DataTrackers::iterator it=m_dataTrackers.begin(),itend=m_dataTrackers.end() ; it!=itend ; ++it )
        it->second = it->first->getCounter();
}

////////////////////
void DataTrackerDDGNode::addInputs(std::initializer_list<sofa::core::objectmodel::BaseData*> datas)
{
    for(sofa::core::objectmodel::BaseData* d : datas) {
        m_dataTracker.trackData(*d);
        addInput(d);
    }
}

void DataTrackerDDGNode::addOutputs(std::initializer_list<sofa::core::objectmodel::BaseData*> datas)
{
    for(sofa::core::objectmodel::BaseData* d : datas)
        addOutput(d);
}

void DataTrackerDDGNode::cleanDirty(const core::ExecParams*)
{
    core::objectmodel::DDGNode::cleanDirty();

    /// it is also time to clean the tracked Data
    m_dataTracker.clean();
}

void DataTrackerDDGNode::updateAllInputsIfDirty()
{
    const DDGLinkContainer& inputs = DDGNode::getInputs();
    for(const auto input : inputs)
    {
        static_cast<core::objectmodel::BaseData*>(input)->updateIfDirty();
    }
}

///////////////////////

void DataTrackerCallback::setCallback( std::function<sofa::core::objectmodel::ComponentState(const DataTracker&)> f)
{
    m_callback = f;
}

void DataTrackerCallback::update()
{
    updateAllInputsIfDirty();

    const auto cs = m_callback(m_dataTracker);
    if (m_owner)
        m_owner->d_componentState.setValue(cs); // but what if the state of the component was invalid for a reason that doesn't depend on this update?
    cleanDirty();
}


}
