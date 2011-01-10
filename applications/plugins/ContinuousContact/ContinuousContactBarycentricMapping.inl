/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_CONTINUOUSCONTACTBARYCENTRICMAPPING_INL
#define SOFA_COMPONENT_MAPPING_CONTINUOUSCONTACTBARYCENTRICMAPPING_INL

#include "ContinuousContactBarycentricMapping.h"

#include <sofa/component/mapping/BarycentricMapping.inl>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
void ContinuousContactBarycentricMapping<TIn, TOut>::beginAddContactPoint()
{
    if (!m_init)
    {
        if (this->mapper)
        {
            this->mapper->clear(0);
        }

        this->toModel->resize(0);

        m_init = true;
    }
    else
        m_init = false;
}

template <class TIn, class TOut>
int ContinuousContactBarycentricMapping<TIn, TOut>::addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & /*baryCoords*/)
{
    if (this->mapper)
    {
        const typename In::VecCoord& xfrom = *this->fromModel->getX();
        typename Out::VecCoord xto;
        xto.resize(1);
        xto[0] = pos;
        this->mapper->init(xfrom, xto);

        /// @TODO get index in mapper
        int index = 0;

        this->toModel->resize(index+1);
        return index;
    }

    return 0;
}

template <class TIn, class TOut>
void ContinuousContactBarycentricMapping<TIn, TOut>::init()
{
    m_init = false;

    Inherit::init();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_CONTINUOUSCONTACTBARYCENTRICMAPPING_INL
