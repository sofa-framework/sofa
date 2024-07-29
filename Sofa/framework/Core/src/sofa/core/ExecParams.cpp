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
#include <sofa/core/ExecParams.h>
#include <sofa/helper/logging/Messaging.h>


namespace sofa::core
{

std::atomic<int> ExecParams::g_nbThreads(0);

ExecParams::ExecParamsThreadStorage::ExecParamsThreadStorage(int tid)
    : execMode(EXEC_DEFAULT)
    , threadID(tid)
{
}

/// Get the default ExecParams, to be used to provide a default values for method parameters
ExecParams* ExecParams::defaultInstance()
{
    thread_local struct ThreadLocalInstance
    {
        ThreadLocalInstance()
            : threadStorage(g_nbThreads.fetch_add(1))
            , params(&threadStorage)
        {
            if (params.threadID())
            {
                msg_info("ExecParams") << "[THREAD " << params.threadID() << "]: local ExecParams storage created.";
            }
        }
        ExecParamsThreadStorage threadStorage;
        ExecParams params;

    } threadParams;
    return &threadParams.params;
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

} // namespace sofa::core


