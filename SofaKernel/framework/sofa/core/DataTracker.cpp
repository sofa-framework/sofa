#include "DataTracker.h"
#include "objectmodel/BaseData.h"

namespace sofa
{

namespace core
{

void DataTracker::setData(objectmodel::BaseData *data, bool dirtyAtBeginning )
{
    this->addInput( (objectmodel::DDGNode*)data );
    if( !dirtyAtBeginning ) this->cleanDirty();
}

const std::string DataTracker::emptyString = "";



//////////////////////


void TrackerDDGNode::updateAllInputsIfDirty()
{
    const DDGLinkContainer& inputs = DDGNode::getInputs();
    for(size_t i=0, iend=inputs.size() ; i<iend ; ++i )
    {
        static_cast<core::objectmodel::BaseData*>(inputs[i])->updateIfDirty();
    }
}

void TrackerDDGNode::cleanDirty(const core::ExecParams* params)
{
    core::objectmodel::DDGNode::cleanDirty(params);

    // it is also time to clean the tracked Data
    for( DataTrackers::iterator it=m_dataTrackers.begin(),itend=m_dataTrackers.end() ; it!=itend ; ++it )
        it->second.cleanDirty();
}

void TrackerDDGNode::trackData( objectmodel::BaseData* data )
{
    m_dataTrackers[data].setData( data );
}

bool TrackerDDGNode::isTrackedDataDirty( const objectmodel::BaseData& data )
{
    return m_dataTrackers[&data].isDirty();
}


}

}
