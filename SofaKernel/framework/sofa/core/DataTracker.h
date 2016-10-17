#ifndef SOFA_CORE_DATATRACKER_H
#define SOFA_CORE_DATATRACKER_H

#include <sofa/core/objectmodel/DDGNode.h>

namespace sofa
{

namespace core
{

    /// Tracking Data mechanism
    struct DataTracker
    {
        /// select a Data to track to be able to check
        /// if it was dirtied since the previous clean.
        /// @see isTrackedDataDirty
        void trackData( const objectmodel::BaseData& data );

        /// Was the tracked Data dirtied since last update?
        /// @warning data must be a tracked Data @see trackData
        bool isDirty( const objectmodel::BaseData& data );

        /// Was one of the tracked Data dirtied since last update?
        bool isDirty();

        /// comparison point is cleaned for the specified tracked Data
        /// @warning data must be a tracked Data @see trackData
        void clean( const objectmodel::BaseData& data );

        /// comparison point is cleaned for all tracked Data
        void clean();


    protected:

        /// map a tracked Data to a DataTracker
        typedef std::map<const objectmodel::BaseData*,int> DataTrackers;
        DataTrackers m_dataTrackers;

    };


//////////////////////////////


    /// A DDGNode with trackable input Data
    class DataTrackerDDGNode : public core::objectmodel::DDGNode
    {
    public:

        DataTrackerDDGNode() : core::objectmodel::DDGNode() {}

    private:
        DataTrackerDDGNode(const DataTrackerDDGNode& n) ;
        DataTrackerDDGNode& operator=(const DataTrackerDDGNode& n) ;

    public:

        /// Set dirty flag to false
        /// for the Engine and for all the tracked Data
        virtual void cleanDirty(const core::ExecParams* params = 0);

    protected:

        /// @name Tracking Data mechanism
        /// each Data added to the DataTracker
        /// is tracked to be able to check if its value changed
        /// since their last clean, called by default
        /// in DataEngine::cleanDirty().
        /// @{

        DataTracker m_dataTracker;

        ///@}

    };


 ///////////////////


    class DataTrackerEngine : DataTrackerDDGNode
    {
    public:

        DataTrackerEngine( const objectmodel::Base* base );

        virtual void updateData() = 0;

    protected:
        const objectmodel::Base* m_base;
    };

} // namespace core

} // namespace sofa

#endif
