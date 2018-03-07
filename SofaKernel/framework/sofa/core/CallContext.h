/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_CALLCONTEXT_H
#define SOFA_CORE_CALLCONTEXT_H

// using smaller header instead of athapascan-1 to try to make compilation faster...
//#include <athapascan-1>
#ifdef SOFA_SMP
#include <kaapi_processor.h>
#endif

#include <sofa/core/core.h>

namespace sofa
{

namespace core
{
#ifdef SOFA_SMP
using namespace Core;
#else
class Processor
{
public:
    enum  ProcessorType {GRAPH_KAAPI,CPU,GPU_CUDA,DEFAULT,VISITOR_SYNC};
    static Processor *get_current();
    inline ProcessorType get_proc_type() {return _proc_type;}
    inline void set_proc_type(ProcessorType _p) {_proc_type=_p;}
    inline unsigned get_pid()
    {
        return 0;
    }
    inline void set_pid(unsigned ) {}
    Processor():_proc_type(DEFAULT) {}
private:
    ProcessorType _proc_type;


};

#endif
class SOFA_CORE_API CallContext
{

public:
    enum  ProcessorType {GRAPH_KAAPI,CPU,GPU_CUDA,DEFAULT,VISITOR_SYNC};
    /// Constructor
    CallContext()
    {

    }
    static ProcessorType  executionType;

    static    inline ProcessorType getProcessorType()
    {
        return	(ProcessorType)Processor::get_current()->get_proc_type();
    }
    static    inline ProcessorType getExecutionType()
    {
        return	(ProcessorType)executionType;
    }
    static inline unsigned getProcessorId()
    {
        return	Processor::get_current()->get_pid();
    }
    static inline void setProcessorType(ProcessorType _type)
    {
        Processor::get_current()->set_proc_type((Processor::ProcessorType)_type);
    }
    static inline void setExecutionType(ProcessorType _type)
    {
        executionType=_type;
    }
    static inline void setProcessorId(unsigned id)
    {
        Processor::get_current()->set_pid(id);
    }

};

} // namespace core

} // namespace sofa

#endif
