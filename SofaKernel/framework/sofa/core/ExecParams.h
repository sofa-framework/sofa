/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_EXEC_PARAMS_H
#define SOFA_CORE_EXEC_PARAMS_H

#include <sofa/helper/system/atomic.h>
#include <sofa/core/core.h>

namespace sofa
{

namespace core
{

#if defined(SOFA_MAX_THREADS)
enum { SOFA_DATA_MAX_ASPECTS = 2*SOFA_MAX_THREADS };
#else
enum { SOFA_DATA_MAX_ASPECTS = 1 };
#endif

#if !defined(NDEBUG) && !defined(SOFA_DEBUG_THREAD)
#define SOFA_DEBUG_THREAD
#endif

/// Class gathering parameters use by most components methods, and transmitted by all visitors
class SOFA_CORE_API ExecParams
{
public:

    /// Modes of execution
    enum ExecMode
    {
        EXEC_NONE = 0,
        EXEC_DEFAULT,
        EXEC_DEBUG,
        EXEC_GPU,
        EXEC_GRAPH
    };

private:

    static sofa::helper::system::atomic<int> g_nbThreads;

    class SOFA_CORE_API ExecParamsThreadStorage
    {
    public:
        /// Mode of execution requested
        ExecMode execMode;

        /// Index of current thread (0 corresponding to the only thread in sequential mode, or first thread in parallel mode)
        int threadID;

        /// Aspect index for the current thread
        int aspectID;

        ExecParamsThreadStorage(int tid);
    };

    static ExecParamsThreadStorage* threadStorage();

    ExecParamsThreadStorage* storage;

    ExecParams(ExecParamsThreadStorage* s)
        : storage(s)
    {
    }


public:
    bool checkValidStorage() const;

    /// Mode of execution requested
    ExecMode execMode() const
    {
#ifdef SOFA_DEBUG_THREAD
        checkValidStorage();
#endif
        return storage->execMode;
    }

    /// Index of current thread (0 corresponding to the only thread in sequential mode, or first thread in parallel mode)
    int threadID() const
    {
#ifdef SOFA_DEBUG_THREAD
        checkValidStorage();
#endif
        return storage->threadID;
    }

    /// Number of threads currently known to Sofa
    int nbThreads() const { return g_nbThreads; }

    /// Aspect index for the current thread
    int aspectID() const
    {
#ifdef SOFA_DEBUG_THREAD
        checkValidStorage();
#endif
        return storage->aspectID;
    }

    ExecParams()
        : storage(threadStorage())
    {
    }

    /// Get the default ExecParams, to be used to provide a default values for method parameters
    static ExecParams* defaultInstance();

    /// Make sure this instance is up-to-date relative to the current thread
    void update();

    /// Request a specific mode of execution
    ExecParams& setExecMode(ExecMode v)
    {
#ifdef SOFA_DEBUG_THREAD
        checkValidStorage();
#endif
        storage->execMode = v;
        return *this;
    }

    /// Specify the index of the current thread
    ExecParams& setThreadID(int v)
    {
#ifdef SOFA_DEBUG_THREAD
        checkValidStorage();
#endif
        storage->threadID = v;
        return *this;
    }

    /// Specify the aspect index of the current thread
    ExecParams& setAspectID(int v)
    {
#ifdef SOFA_DEBUG_THREAD
        checkValidStorage();
#endif
        storage->aspectID = v;
        return *this;
    }

    static int currentAspect()
    {
        if (SOFA_DATA_MAX_ASPECTS == 1) return 0;
        else                             return defaultInstance()->aspectID();
    }

    static inline int currentAspect(const core::ExecParams* params)
    {
        if (SOFA_DATA_MAX_ASPECTS == 1) return 0;
        else                             return params != 0 ? params->aspectID() : defaultInstance()->aspectID();
    }

};

} // namespace core

} // namespace sofa

#endif
