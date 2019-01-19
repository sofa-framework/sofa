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
#ifndef ParallelFor2d_h__
#define ParallelFor2d_h__

#include <sofa/config.h>

#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/ParallelFor.h>

namespace sofa
{
	namespace simulation
	{

        class ForTask2d
        {
        public:

            class Range
            {
            public:

                Range() = default;

                Range(ForTask::Range rows, ForTask::Range cols)
                    : _rows(rows)
                    , _cols(cols)
                {}

                const ForTask::Range& rows() const { return _rows; }
                const ForTask::Range& cols() const { return _cols; }

                static ForTask2d::Range split(ForTask2d::Range& r)
                { 
                    if (r._rows.size()*r._cols.grainsize() < r._cols.size()*r._rows.grainsize()) 
                    {
                        return ForTask2d::Range(r._rows, ForTask::Range::split(r._cols) );
                        //_cols._first = col_range_type::do_split(r._cols);
                    }
                    else
                    {
                        return ForTask2d::Range(ForTask::Range::split(r._rows), r._cols);
                    }
                }

                bool is_divisible() const { return _rows.is_divisible() || _cols.is_divisible(); }

            private:

                ForTask::Range _rows;
                ForTask::Range _cols;

                friend class InternalForTask2d;
            };


            enum Partition
            {
                simple = 0,
                avoid_shared_data = 1
            };

        public:

            ForTask2d() {}

            virtual ~ForTask2d() {}

            virtual void operator()(const ForTask2d::Range& range) const = 0;

        };



        SOFA_SIMULATION_CORE_API void ParallelFor2d(ForTask2d& task, const ForTask2d::Range& range, ForTask2d::Partition partition = ForTask2d::Partition::simple);


	} // namespace simulation

} // namespace sofa


#endif // ParallelFor2d_h__
