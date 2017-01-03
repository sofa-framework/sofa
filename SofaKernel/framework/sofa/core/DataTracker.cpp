#include "DataTracker.h"
#include "objectmodel/BaseData.h"

namespace sofa
{

namespace core
{





void DataTracker::trackData( const objectmodel::BaseData& data )
{
    m_dataTrackers[&data] = data.getCounter();
}

bool DataTracker::wasModified( const objectmodel::BaseData& data ) const
{
    return m_dataTrackers.at(&data) != data.getCounter();
}

bool DataTracker::wasModified() const
{
    for( DataTrackers::const_iterator it=m_dataTrackers.begin(),itend=m_dataTrackers.end() ; it!=itend ; ++it )
        if( it->second != it->first->getCounter() ) return true;
    return false;
}

bool DataTracker::isDirty( const objectmodel::BaseData& data ) const
{
    return wasModified(data) || data.isDirty();
}

bool DataTracker::isDirty() const
{
    for( DataTrackers::const_iterator it=m_dataTrackers.begin(),itend=m_dataTrackers.end() ; it!=itend ; ++it )
        if( it->second != it->first->getCounter() || it->first->isDirty() ) return true;
    return false;
}


void DataTracker::clean( const objectmodel::BaseData& data )
{
    m_dataTrackers.at(&data) = data.getCounter();
}

void DataTracker::clean()
{
    for( DataTrackers::iterator it=m_dataTrackers.begin(),itend=m_dataTrackers.end() ; it!=itend ; ++it )
        it->second = it->first->getCounter();
}



////////////////////



void DataTrackerDDGNode::cleanDirty(const core::ExecParams* params)
{
    core::objectmodel::DDGNode::cleanDirty(params);

    // it is also time to clean the tracked Data
    m_dataTracker.clean();

}



void DataTrackerDDGNode::updateAllInputsIfDirty()
{
    const DDGLinkContainer& inputs = DDGNode::getInputs();
    for(size_t i=0, iend=inputs.size() ; i<iend ; ++i )
    {
        static_cast<core::objectmodel::BaseData*>(inputs[i])->updateIfDirty();
    }
}



///////////////////////


void DataTrackerEngine::setUpdateCallback( void (*f)(DataTrackerEngine*) )
{
    m_updateCallback = f;
}

}

}
