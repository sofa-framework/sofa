/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef ParallelFor_h__
#define ParallelFor_h__

#include <sofa/config.h>

#include <sofa/simulation/TaskScheduler.h>


namespace sofa
{
	namespace simulation
	{

        class ForTask
        {
        public:

            class Range
            {
            public:

                Range() = default;

                Range(uint64_t first, uint64_t last, uint64_t grainsize = 1)
                    : _first(first)
                    , _last(last)
                    , _grainsize(grainsize)
                {}

                uint64_t first() const { return _first; }
                uint64_t last() const { return _last; }
                uint64_t grainsize() const { return _grainsize; }

                static Range split(Range& r);

                uint64_t size() const
                {
                    assert(!(_last < _first));
                    return _last - _first;
                }

                bool is_divisible() const { return _grainsize<size(); }

            private:

                uint64_t _first;
                uint64_t _last;
                uint64_t _grainsize;
            };


        public:

            ForTask() {}

            virtual ~ForTask() {}

            virtual void operator()(const ForTask::Range& range) const = 0;

        };


        SOFA_SIMULATION_CORE_API void ParallelFor(ForTask& task, const ForTask::Range& range);



	} // namespace simulation

} // namespace sofa


#endif // ParallelFor_h__
