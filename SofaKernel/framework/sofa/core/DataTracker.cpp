#include "DataTracker.h"

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



}

}
