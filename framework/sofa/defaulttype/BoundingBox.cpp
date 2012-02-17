/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/defaulttype/BoundingBox.h>
#include <limits>

namespace sofa
{
namespace defaulttype
{


namespace
{
BoundingBox::bbox_t make_neutralBBox()
{
    typedef sofa::defaulttype::Vector3::value_type Real;
    const Real max_real = std::numeric_limits<Real>::max();
    sofa::defaulttype::Vector3 minBBox(max_real,max_real,max_real);
    sofa::defaulttype::Vector3 maxBBox(-max_real,-max_real,-max_real);
    return std::make_pair(minBBox,maxBBox);
}
}

BoundingBox::BoundingBox()
    :bbox(make_neutralBBox())
{
}

BoundingBox::BoundingBox(const bbox_t& bbox)
    :bbox(bbox)
{
}

BoundingBox::BoundingBox(const Vector3& minBBox, const Vector3& maxBBox)
    :bbox(std::make_pair(minBBox,maxBBox))
{
}

BoundingBox::BoundingBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax )
    :bbox(std::make_pair(Vector3(xmin, ymin,zmin),Vector3(xmax,ymax,zmax)))
{
}


BoundingBox::BoundingBox(const Vec6f& v )
    :bbox(std::make_pair(Vector3(v[0],v[2],v[4]),Vector3(v[1],v[3],v[5])))
{
}

BoundingBox::BoundingBox(const Vec6d& v )
    :bbox(std::make_pair(Vector3(v[0],v[2],v[4]),Vector3(v[1],v[3],v[5])))
{
}


/*static*/
BoundingBox BoundingBox::neutral_bbox()
{
    return BoundingBox(make_neutralBBox());
}

void BoundingBox::invalidate()
{
    this->bbox = make_neutralBBox();
}

bool BoundingBox::isValid() const
{
    return minBBox().x() <= maxBBox().x() &&
            minBBox().y() <= maxBBox().y() &&
            minBBox().z() <= maxBBox().z();
}

bool BoundingBox::isFlat() const
{
    return    minBBox().x() == maxBBox().x() ||
            minBBox().y() == maxBBox().y() ||
            minBBox().z() == maxBBox().z();
}

BoundingBox::operator bbox_t() const
{
    return bbox;
}

SReal* BoundingBox::minBBoxPtr()
{
    return bbox.first.elems;
}

SReal* BoundingBox::maxBBoxPtr()
{
    return bbox.second.elems;
}

const SReal* BoundingBox::minBBoxPtr() const
{
    return bbox.first.elems;
}

const SReal* BoundingBox::maxBBoxPtr() const
{
    return bbox.second.elems;
}

const Vector3& BoundingBox::minBBox() const
{
    return bbox.first;
}

const Vector3& BoundingBox::maxBBox() const
{
    return bbox.second;
}

Vector3& BoundingBox::minBBox()
{
    return bbox.first;
}

Vector3& BoundingBox::maxBBox()
{
    return bbox.second;
}

bool BoundingBox::contains(const sofa::defaulttype::Vector3& point) const
{
    return  point.x() >= minBBox().x() && point.x() <= maxBBox().x() &&
            point.y() >= minBBox().y() && point.y() <= maxBBox().y() &&
            point.z() >= minBBox().z() && point.z() <= maxBBox().z();
}

bool BoundingBox::contains(const BoundingBox& other) const
{
    return contains(other.minBBox()) && contains(other.maxBBox());
}

bool BoundingBox::intersect(const BoundingBox& other) const
{
    if( other.minBBox().x() > maxBBox().x() || other.maxBBox().x ()< minBBox().x() ) return false;
    if( other.minBBox().y() > maxBBox().y() || other.maxBBox().y ()< minBBox().y() ) return false;
    if( other.minBBox().z() > maxBBox().z() || other.maxBBox().z ()< minBBox().z() ) return false;
    return true;
}

void BoundingBox::intersection(const BoundingBox& other)
{
    minBBox().x() = std::max(minBBox().x(), other.minBBox().x());
    minBBox().y() = std::max(minBBox().y(), other.minBBox().y());
    minBBox().z() = std::max(minBBox().z(), other.minBBox().z());

    maxBBox().x() = std::min(maxBBox().x(), other.maxBBox().x());
    maxBBox().y() = std::min(maxBBox().y(), other.maxBBox().y());
    maxBBox().z() = std::min(maxBBox().z(), other.maxBBox().z());
}

void BoundingBox::include(const sofa::defaulttype::Vector3& point)
{
    minBBox().x() = std::min( minBBox().x(), point.x());
    minBBox().y() = std::min( minBBox().y(), point.y());
    minBBox().z() = std::min( minBBox().z(), point.z());

    maxBBox().x() = std::max( maxBBox().x(), point.x());
    maxBBox().y() = std::max( maxBBox().y(), point.y());
    maxBBox().z() = std::max( maxBBox().z(), point.z());
}

void BoundingBox::include(const BoundingBox& other)
{
    minBBox().x() = std::min( minBBox().x(), other.minBBox().x());
    minBBox().y() = std::min( minBBox().y(), other.minBBox().y());
    minBBox().z() = std::min( minBBox().z(), other.minBBox().z());

    maxBBox().x() = std::max( maxBBox().x(), other.maxBBox().x());
    maxBBox().y() = std::max( maxBBox().y(), other.maxBBox().y());
    maxBBox().z() = std::max( maxBBox().z(), other.maxBBox().z());
}

void BoundingBox::inflate(const SReal amount)
{
    Vector3 size(amount,amount,amount);
    minBBox() -= size;
    maxBBox() += size;
}

} // namespace defaulttype

} // namespace sofa
