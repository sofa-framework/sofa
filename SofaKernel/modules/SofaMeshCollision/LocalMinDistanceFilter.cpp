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

#include <SofaMeshCollision/LocalMinDistanceFilter.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaRigid/RigidMapping.h>
#include <limits>


namespace sofa
{

namespace component
{

namespace collision
{
using namespace defaulttype;
using namespace core::behavior;
using namespace sofa::component::mapping;


LocalMinDistanceFilter::LocalMinDistanceFilter()
    : m_coneExtension(initData(&m_coneExtension, 0.5, "coneExtension", "Filtering cone extension angle."))
    , m_coneMinAngle(initData(&m_coneMinAngle, 0.0, "coneMinAngle", "Minimal filtering cone angle value, independent from geometry."))
    , m_revision(0)
    , m_rigid(initData(&m_rigid, false, "isRigid", "filters optimization for rigid case."))
{
}



LocalMinDistanceFilter::~LocalMinDistanceFilter()
{
}

// when object is rigid, bwdInit will look for the RigidMapping
// to get a point on the MState of the rigid frame
void LocalMinDistanceFilter::bwdInit()
{
    if(this->isRigid())
    {
        RigidMapping< Rigid3Types, Vec3Types > *r_mapping= NULL;
        r_mapping = this->getContext()->get< RigidMapping< Rigid3Types, Vec3Types > >();

        if(r_mapping==NULL)
        {
            serr<<"No RigidMapping were found in the same or child node: maybe a template problem (only works for double)"<<sendl;
            this->setRigid(false);
            return;
        }
        if(!r_mapping->useX0.getValue())
        {
            serr<<"optimization for rigid can not be used if the RigidMapping.useX0=false : cancel optim"<<sendl;
            this->setRigid(false);
            return;
        }
    }
}


// invalidate function is called by PointModel, LineModel or TriangleModel each time a new computation of the BoundingTree is called
void LocalMinDistanceFilter::invalidate()
{

    if(this->isRigid())
        return;         // If the object is rigid, the filters are pre-built

    /// TODO: this does not do anything...
    if (m_revision >= std::numeric_limits< unsigned int >::max())
        m_revision=0;
}

bool InfoFilter::isValid(void)
{
    assert(m_lmdFilters != 0);
    if (m_lmdFilters==NULL)
    {
        msg_error("InfoFilter")<<"pointer m_lmdFilters is null";
        return false;
    }

    return m_revision == m_lmdFilters->getRevision();
}

void InfoFilter::setValid()
{
    assert(m_lmdFilters != 0);
    m_revision = m_lmdFilters->getRevision();
}

} // namespace collision

} // namespace component

} // namespace sofa
