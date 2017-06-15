/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/core/ExecParams.h>
#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace core
{

sofa::helper::system::atomic<int> ExecParams::g_nbThreads = 0;

ExecParams::ExecParamsThreadStorage::ExecParamsThreadStorage(int tid)
    : execMode(EXEC_DEFAULT)
    , threadID(tid)
    , aspectID(0)
{
}

/// Get the default ExecParams, to be used to provide a default values for method parameters
ExecParams* ExecParams::defaultInstance()
{
    SOFA_THREAD_SPECIFIC_PTR(ExecParams, threadParams);
    ExecParams* ptr = threadParams;
    if (!ptr)
    {
        ptr = new ExecParams(new ExecParamsThreadStorage(g_nbThreads.exchange_and_add(1)));
        threadParams = ptr;
        if (ptr->threadID())
            msg_info("ExecParams") << "[THREAD " << ptr->threadID() << "]: local ExecParams storage created.";
    }
    return ptr;
}

ExecParams::ExecParamsThreadStorage* ExecParams::threadStorage()
{
    return defaultInstance()->storage;
}

bool ExecParams::checkValidStorage() const
{
    ExecParams::ExecParamsThreadStorage* ts = threadStorage();

    if (storage == ts)
        return true;

    msg_error("ExecParams") <<  "[THREAD " << ts->threadID << "]:  invalid ExecParams used, belonging to thread " << storage->threadID;
    const_cast<ExecParams*>(this)->storage = ts;
    return false;
}

/// Make sure this instance is up-to-date relative to the current thread
void ExecParams::update()
{
    storage = threadStorage();
}

} // namespace core

} // namespace sofa
