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

#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/task/CpuTaskStatus.h>
#include <sofa/type/vector_T.h>

namespace sofa::simulation
{

/**
 * Represents an iterable sequence in a container
 */
template<class InputIt>
struct Range
{
    InputIt start;
    InputIt end;

    Range(InputIt s, InputIt e) : start(s), end(e) {}
};

template<class InputIt, class Distance>
void advance(InputIt& it, Distance n)
{
    if constexpr (std::is_integral_v<InputIt>)
    {
        it += n;
    }
    else
    {
        std::advance(it, n);
    }
}

/**
 * Function returning a list of ranges from an iterable container.
 * The number of ranges depends on:
 *  1) the desired number of ranges provided in a parameter
 *  2) the number of elements in the container
 * The number of elements in each range is homogeneous, except for the last range which may contain
 * more elements.
 */
template<class InputIt>
sofa::type::vector<Range<InputIt> >
makeRangesForLoop(const InputIt first, const InputIt last, const unsigned int nbRangesHint)
{
    sofa::type::vector<Range<InputIt> > ranges;

    if (first == last)
    {
        return ranges;
    }

    unsigned int nbElements = 0;
    if constexpr (std::is_integral_v<InputIt>)
    {
        nbElements = static_cast<unsigned int>(last - first);
    }
    else
    {
        nbElements = static_cast<unsigned int>(std::distance(first, last));
    }

    const unsigned int nbRanges = std::min(nbRangesHint, nbElements);
    ranges.reserve(nbRanges);

    const auto nbElementsPerRange = nbElements / nbRanges;

    Range<InputIt> r { first, first};
    sofa::simulation::advance(r.end, nbElementsPerRange);

    for (unsigned int i = 0; i < nbRanges - 1; ++i)
    {
        ranges.emplace_back(r);

        sofa::simulation::advance(r.start, nbElementsPerRange);
        sofa::simulation::advance(r.end, nbElementsPerRange);
    }

    ranges.emplace_back(r.start, last);

    return ranges;
}

/**
 * Applies the given function object f to the result of dereferencing every iterator in the
 * range [first, last), in order.
 */
template<class InputIt, class UnaryFunction>
UnaryFunction forEach(InputIt first, InputIt last, UnaryFunction f)
{
    if constexpr (std::is_integral_v<InputIt>)
    {
        for (; first != last; ++first)
        {
            f(first);
        }
        return f;
    }
    else
    {
        return std::for_each(first, last, f);
    }
}

/**
 * Applies the given function object f to the Range [first, last)
 *
 * The signature of the function f should be equivalent to the following:
 * void fun(const Range<InputIt>& a);
 * The signature does not need to have const &
 */
template<class InputIt, class UnaryFunction>
UnaryFunction forEachRange(InputIt first, InputIt last, UnaryFunction f)
{
    Range<InputIt> r{ first, last};
    f(r);

    return f;
}

/**
 * Applies in parallel the given function object f to a list of ranges generated from [first, last)
 *
 * The signature of the function f should be equivalent to the following:
 * void fun(const Range<InputIt>& a);
 * The signature does not need to have const &.
 *
 * A task scheduler must be provided and correctly initialized. The number of generated ranges
 * depends on the threads available in the task scheduler.
 */
template<class InputIt, class UnaryFunction>
UnaryFunction parallelForEachRange(TaskScheduler& taskScheduler, InputIt first, InputIt last, UnaryFunction f)
{
    if (first != last)
    {
        const auto taskSchedulerThreadCount = taskScheduler.getThreadCount();
        if (taskSchedulerThreadCount == 0)
        {
            msg_error("parallelForEach") << "Task scheduler does not appear to be initialized. Cannot perform parallel tasks.";
            return forEachRange(first, last, f);
        }

        CpuTaskStatus status;

        const auto ranges = makeRangesForLoop<InputIt>(first, last, taskSchedulerThreadCount);

        for (const Range<InputIt>& r : ranges)
        {
            taskScheduler.addTask(status, [&r, &f]()
            {
                f(r);
            });
        }

        taskScheduler.workUntilDone(&status);
    }
    return f;
}

/**
 * Applies the given function object f to the result of dereferencing every iterator in the
 * range [first, last), in parallel.
 */
template<class InputIt, class UnaryFunction>
UnaryFunction parallelForEach(TaskScheduler& taskScheduler, InputIt first, InputIt last, UnaryFunction f)
{
    parallelForEachRange(taskScheduler, first, last,
        [&f](const Range<InputIt>& r)
        {
            forEach(r.start, r.end, f);
        });
    return f;
}


enum class ForEachExecutionPolicy : bool
{
    SEQUENTIAL = false,
    PARALLEL
};

template<class InputIt, class UnaryFunction>
UnaryFunction forEachRange(const ForEachExecutionPolicy execution, TaskScheduler& taskScheduler,
                      InputIt first,
                      InputIt last, UnaryFunction f)
{
    if (execution == ForEachExecutionPolicy::PARALLEL)
    {
        return parallelForEachRange(taskScheduler, first, last, f);
    }
    return forEachRange(first, last, f);
}

template<class InputIt, class UnaryFunction>
UnaryFunction forEach(const ForEachExecutionPolicy execution, TaskScheduler& taskScheduler,
                      InputIt first,
                      InputIt last, UnaryFunction f)
{
    if (execution == ForEachExecutionPolicy::PARALLEL)
    {
        return parallelForEach(taskScheduler, first, last, f);
    }
    return forEach(first, last, f);
}

}
