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




    /// A DDGNode with trackable input Data
    class TrackerDDGNode : public core::objectmodel::DDGNode
    {
    public:

        TrackerDDGNode() : core::objectmodel::DDGNode() {}

    protected:

        /// utility fonction to ensure all inputs are up-to-date
        /// can be useful for particulary complex DataEngine
        /// with a lot input/output imbricated access
        void updateAllInputsIfDirty();

    private:
        TrackerDDGNode(const TrackerDDGNode& n) ;
        TrackerDDGNode& operator=(const TrackerDDGNode& n) ;

    public:

        /// Set dirty flag to false
        /// for the Engine and for all the tracked Data
        virtual void cleanDirty(const core::ExecParams* params = 0);

    protected:


        /// @name Tracking Data mechanism
        /// each tracked Data is connected to a DataTracker
        /// that is dirtied with the tracked Data
        /// but cleaned only in the DataEngine::cleanDirty()
        /// @{

        /// select a Data to track to be able to check
        /// if it was dirtied since the previous update.
        /// @see isTrackedDataDirty
        void trackData( objectmodel::BaseData* data );

        /// Was the tracked Data dirtied since last update?
        /// @warning data must be a tracked Data @see trackData
        bool isTrackedDataDirty( const objectmodel::BaseData& data );

        /// map a tracked Data to a DataTracker
        typedef std::map<const objectmodel::BaseData*,DataTracker> DataTrackers;
        DataTrackers m_dataTrackers;

        /// @}

    };

} // namespace core

} // namespace sofa

#endif
