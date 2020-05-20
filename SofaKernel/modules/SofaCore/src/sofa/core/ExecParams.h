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
#ifndef SOFA_CORE_EXEC_PARAMS_H
#define SOFA_CORE_EXEC_PARAMS_H

#include <sofa/core/core.h>
#include <atomic>

namespace sofa
{

namespace core
{


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

    static std::atomic<int> g_nbThreads;

    class SOFA_CORE_API ExecParamsThreadStorage
    {
    public:
        /// Mode of execution requested
        ExecMode execMode;

        /// Index of current thread (0 corresponding to the only thread in sequential mode, or first thread in parallel mode)
        int threadID;

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

    ////////////////////////////////////// DEPRECATED ///////////////////////////////////////////
    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. If the feature was important to you contact sofa-dev. ")]]
    int aspectID() const { return 0; }

    /// Specify the aspect index of the current thread
    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. If the feature was important to you contact sofa-dev. ")]]
    ExecParams& setAspectID(int /* v */){ return *this; }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. If the feature was important to you contact sofa-dev. ")]]
    static int currentAspect(){ return 0; }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. If the feature was important to you contact sofa-dev. ")]]
    static int currentAspect(const core::ExecParams*){ return 0; }
};

} // namespace core

} // namespace sofa

#endif
