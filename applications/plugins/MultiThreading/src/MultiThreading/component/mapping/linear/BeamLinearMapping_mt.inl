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
#pragma once

#include <MultiThreading/component/mapping/linear/BeamLinearMapping_mt.h>

#include <MultiThreading/component/mapping/linear/BeamLinearMapping_tasks.inl>

#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>

namespace multithreading::component::mapping::linear
{
    using namespace sofa::defaulttype;

    template <class TIn, class TOut>
    BeamLinearMapping_mt< TIn, TOut>::BeamLinearMapping_mt()
    : mGrainSize(initData(&mGrainSize, (unsigned int)32,"granularity", "minimum number of Beam points for task creation" ))
    {
    }
    
    
    template <class TIn, class TOut>
    BeamLinearMapping_mt< TIn, TOut>::~BeamLinearMapping_mt()
    {
    }
    
    
    template <class TIn, class TOut>
    void BeamLinearMapping_mt< TIn, TOut>::init()
    {
        sofa::simulation::MainTaskSchedulerFactory::createInRegistry()->init();
        
        sofa::component::mapping::linear::BeamLinearMapping< TIn, TOut>::init();
    }
    
    template <class TIn, class TOut>
    void BeamLinearMapping_mt< TIn, TOut>::bwdInit()
    {
        sofa::component::mapping::linear::BeamLinearMapping< TIn, TOut>::bwdInit();
    }
    
    
    template <class TIn, class TOut>
    void BeamLinearMapping_mt< TIn, TOut>::apply(const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, sofa::Data<VecCoord>& _out, const sofa::Data<typename In::VecCoord>& _in)
    {
        
        //Inherit::apply(mparams, dOut, dIn);
        //boost::pool<> task_pool(sizeof(BeamLinearMapping_mt< TIn, TOut>::applyTask));
        
        unsigned int numPoints = this->points.size();
        
        if ( numPoints >  2*mGrainSize.getValue()  )
        {
            sofa::helper::WriteAccessor< sofa::Data< typename Out::VecCoord > > out = _out;
            sofa::helper::ReadAccessor< sofa::Data< typename In::VecCoord > > in = _in;
            
            //const InVecCoord& in= _in.getValue();
            //VecCoord& out = *_out.beginEdit();
            
            this->rotatedPoints0.resize(this->points.size());
            this->rotatedPoints1.resize(this->points.size());
            out.resize(this->points.size());
            
            
            // create tasks
            sofa::simulation::CpuTask::Status status;
            sofa::simulation::TaskScheduler* scheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
            
            const int taskSize = 2*mGrainSize.getValue();
            
            int nbTasks = numPoints / taskSize;
            int pointsLeft = numPoints % taskSize;
            
            for ( int i=0; i<nbTasks; ++i)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyTask( &status );
                
                task->_mapping = this;
                //task->_mparams = mparams;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = i*taskSize;
                task->_lastPoint = i*taskSize + mGrainSize.getValue();
                
                scheduler->addTask( task );
                
            }
            if ( pointsLeft > 0)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyTask( &status );
                
                task->_mapping = this;
                //task->_mparams = mparams;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = nbTasks*taskSize;
                task->_lastPoint = nbTasks*taskSize + pointsLeft;
                
                scheduler->addTask( task );
                
            }
            
            scheduler->workUntilDone(&status);
            
            
            for ( int i=0; i<nbTasks; ++i)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyTask( &status );
                
                task->_mapping = this;
                //task->_mparams = mparams;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = i*taskSize + mGrainSize.getValue();
                task->_lastPoint = i*taskSize + taskSize;
                
                scheduler->addTask( task );
                
            }
            
            scheduler->workUntilDone(&status);
            
        }
        else
        {
            
            sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::apply( mparams, _out, _in );
            
        }
        
        // it doesn't call the destructor
        //task_pool.purge_memory();
        
    }
    
    
    
    template <class TIn, class TOut>
    //void AdaptiveBeamMapping< TIn, TOut>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
    void BeamLinearMapping_mt< TIn, TOut>::applyJ(const sofa::core::MechanicalParams * params /* PARAMS FIRST */, sofa::Data< typename Out::VecDeriv >& _out, const sofa::Data< typename In::VecDeriv >& _in)
    {
        
        //boost::pool<> task_pool(sizeof(BeamLinearMapping_mt< TIn, TOut>::applyJTask));
        unsigned int numPoints = this->points.size();
        
        if ( numPoints >  2*mGrainSize.getValue()  )
        {
            sofa::helper::WriteAccessor< sofa::Data< typename Out::VecDeriv > > out = _out;
            sofa::helper::ReadAccessor< sofa::Data< typename In::VecDeriv > > in = _in;
            
            //const InVecDeriv& in= dIn.getValue();
            //VecDeriv& out = *dOut.beginEdit();
            
            
            out.resize(this->points.size());
            
            sofa::simulation::CpuTask::Status status;
            sofa::simulation::TaskScheduler* scheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
            
            const int taskSize = 2*mGrainSize.getValue();
            
            int nbTasks = numPoints / taskSize;
            int pointsLeft = numPoints % taskSize;
            
            for ( int i=0; i<nbTasks; ++i)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyJTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyJTask( &status );
                
                task->_mapping = this;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = i*taskSize;
                task->_lastPoint = i*taskSize + mGrainSize.getValue();
                
                scheduler->addTask( task );
                
            }
            if ( pointsLeft > 0)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyJTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyJTask( &status );
                
                task->_mapping = this;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = nbTasks*taskSize;
                task->_lastPoint = nbTasks*taskSize + pointsLeft;
                
                scheduler->addTask( task );
                
            }
            
            scheduler->workUntilDone(&status);
            
            
            for ( int i=0; i<nbTasks; ++i)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyJTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyJTask( &status );
                
                task->_mapping = this;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = i*taskSize + mGrainSize.getValue();
                task->_lastPoint = i*taskSize + taskSize;
                
                scheduler->addTask( task );
                
            }
            
            scheduler->workUntilDone(&status);
            
        }
        else
        {
            
            sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::applyJ( params, _out, _in );
            
        }
        
        // it doesn't call the destructor
        //task_pool.purge_memory();
        
    }
    
    
    
    
    template <class TIn, class TOut>
    void BeamLinearMapping_mt<TIn, TOut>::applyJT(const sofa::core::MechanicalParams * mparams /* PARAMS FIRST */, sofa::Data< typename In::VecDeriv >& _out, const sofa::Data< typename Out::VecDeriv >& _in)
    {
        
        //boost::pool<> task_pool(sizeof(BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask));
        
        unsigned int numPoints = this->points.size();
        
        if ( numPoints >  2*mGrainSize.getValue()  )
        {
            sofa::helper::WriteAccessor< sofa::Data< typename In::VecDeriv > > out = _out;
            sofa::helper::ReadAccessor< sofa::Data< typename Out::VecDeriv > > in = _in;
            
            
            sofa::simulation::CpuTask::Status status;
            sofa::simulation::TaskScheduler* scheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
            
            const int taskSize = 2*mGrainSize.getValue();
            
            int nbTasks = numPoints / taskSize;
            int pointsLeft = numPoints % taskSize;
            
            for ( int i=0; i<nbTasks; ++i)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask( &status );
                
                task->_mapping = this;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = i*taskSize;
                task->_lastPoint = i*taskSize + mGrainSize.getValue();
                
                scheduler->addTask( task );
                
            }
            if ( pointsLeft > 0)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask( &status );
                
                task->_mapping = this;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = nbTasks*taskSize;
                task->_lastPoint = nbTasks*taskSize + pointsLeft;
                
                scheduler->addTask( task );
                
            }
            
            scheduler->workUntilDone(&status);
            
            
            for ( int i=0; i<nbTasks; ++i)
            {
                typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask* task =
                new typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask( &status );
                
                task->_mapping = this;
                task->_in = &in;
                task->_out = &out;
                task->_firstPoint = i*taskSize + mGrainSize.getValue();
                task->_lastPoint = i*taskSize + taskSize;
                
                scheduler->addTask( task );
                
            }
            
            scheduler->workUntilDone(&status);
            
        }
        else
        {
            
            sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::applyJT( mparams, _out, _in );
            
        }
        
        // it doesn't call the destructor
        //task_pool.purge_memory();
        
    }
} // namespace sofa::component::mapping

