#ifndef SOFA_CORE_DATATRACKER_H
#define SOFA_CORE_DATATRACKER_H

#include <sofa/core/objectmodel/DDGNode.h>

namespace sofa
{

namespace core
{

    /// An empty DDGNode to track if a Data changed.
    /// Each time the corresponding Data changes, the DataTracker will be dirty.
    class DataTracker : public objectmodel::DDGNode
    {
    public:
        /// set the Data to track
        void setData( objectmodel::BaseData* data, bool dirtyAtBeginning=false );
    private:
        static const std::string emptyString;
        virtual void update() { cleanDirty(); }
        virtual const std::string& getName() const { return emptyString; }
        virtual objectmodel::Base* getOwner() const { return NULL; }
        virtual objectmodel::BaseData* getData() const { return NULL; }
    };

} // namespace core

} // namespace sofa

#endif
