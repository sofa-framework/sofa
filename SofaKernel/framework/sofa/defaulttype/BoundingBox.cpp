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

BoundingBox::BoundingBox(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax )
    :bbox(std::make_pair(Vector3((SReal)xmin, (SReal)ymin, (SReal)zmin),Vector3( (SReal)xmax, (SReal)ymax, (SReal)zmax)))
{
}


BoundingBox::BoundingBox(const Vec6f& v )
    :bbox(std::make_pair(Vector3(v[0],v[2],v[4]),Vector3(v[1],v[3],v[5])))
{
}

BoundingBox::BoundingBox(const Vec6d& v )
    :bbox(std::make_pair(Vector3((SReal)v[0],(SReal)v[2],(SReal)v[4]),Vector3((SReal)v[1],(SReal)v[3],(SReal)v[5])))
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

bool BoundingBox::isNegligeable() const
{
    return minBBox().x() >= maxBBox().x() &&
           minBBox().y() >= maxBBox().y() &&
           minBBox().z() >= maxBBox().z();
}

bool BoundingBox::isValid() const
{
    return minBBox().x() <= maxBBox().x() &&
           minBBox().y() <= maxBBox().y() &&
           minBBox().z() <= maxBBox().z();
}

bool BoundingBox::isFlat() const
{
    return minBBox().x() == maxBBox().x() ||
           minBBox().y() == maxBBox().y() ||
           minBBox().z() == maxBBox().z();
}

bool BoundingBox::isNull() const
{
    return minBBox().x() == maxBBox().x() &&
           minBBox().y() == maxBBox().y() &&
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


BoundingBox BoundingBox::getIntersection( const BoundingBox& other ) const
{
    BoundingBox result;

    result.minBBox().x() = std::max(minBBox().x(), other.minBBox().x());
    result.minBBox().y() = std::max(minBBox().y(), other.minBBox().y());
    result.minBBox().z() = std::max(minBBox().z(), other.minBBox().z());

    result.maxBBox().x() = std::min(maxBBox().x(), other.maxBBox().x());
    result.maxBBox().y() = std::min(maxBBox().y(), other.maxBBox().y());
    result.maxBBox().z() = std::min(maxBBox().z(), other.maxBBox().z());

    return result;
}

BoundingBox BoundingBox::getInclude( const sofa::defaulttype::Vector3& point ) const
{
    BoundingBox result;

    result.minBBox().x() = std::min( minBBox().x(), point.x());
    result.minBBox().y() = std::min( minBBox().y(), point.y());
    result.minBBox().z() = std::min( minBBox().z(), point.z());

    result.maxBBox().x() = std::max( maxBBox().x(), point.x());
    result.maxBBox().y() = std::max( maxBBox().y(), point.y());
    result.maxBBox().z() = std::max( maxBBox().z(), point.z());

    return result;
}

BoundingBox BoundingBox::getInclude( const BoundingBox& other ) const
{
    BoundingBox result;

    result.minBBox().x() = std::min( minBBox().x(), other.minBBox().x());
    result.minBBox().y() = std::min( minBBox().y(), other.minBBox().y());
    result.minBBox().z() = std::min( minBBox().z(), other.minBBox().z());

    result.maxBBox().x() = std::max( maxBBox().x(), other.maxBBox().x());
    result.maxBBox().y() = std::max( maxBBox().y(), other.maxBBox().y());
    result.maxBBox().z() = std::max( maxBBox().z(), other.maxBBox().z());

    return result;
}

BoundingBox BoundingBox::getInflate( SReal amount ) const
{
    BoundingBox result;

    Vector3 size(amount,amount,amount);
    result.minBBox() = minBBox() - size;
    result.maxBBox() = maxBBox() + size;

    return result;
}



//////////


namespace
{
BoundingBox2D::bbox_t make_neutralBBox2D()
{
    typedef sofa::defaulttype::Vector2::value_type Real;
    const Real max_real = std::numeric_limits<Real>::max();
    sofa::defaulttype::Vector2 minBBox(max_real,max_real);
    sofa::defaulttype::Vector2 maxBBox(-max_real,-max_real);
    return std::make_pair(minBBox,maxBBox);
}
}

BoundingBox2D::BoundingBox2D()
    :bbox(make_neutralBBox2D())
{
}

BoundingBox2D::BoundingBox2D(const bbox_t& bbox)
    :bbox(bbox)
{
}

BoundingBox2D::BoundingBox2D(const Vector2& minBBox, const Vector2& maxBBox)
    :bbox(std::make_pair(minBBox,maxBBox))
{
}

BoundingBox2D::BoundingBox2D(SReal xmin, SReal xmax, SReal ymin, SReal ymax )
    :bbox(std::make_pair(Vector2((SReal)xmin, (SReal)ymin),Vector2((SReal)xmax,(SReal)ymax)))
{
}


BoundingBox2D::BoundingBox2D(const Vec4f& v )
    :bbox(std::make_pair(Vector2(v[0],v[2]),Vector2(v[1],v[3])))
{
}

BoundingBox2D::BoundingBox2D(const Vec4d& v )
    :bbox(std::make_pair(Vector2((SReal)v[0],(SReal)v[2]),Vector2((SReal)v[1],(SReal)v[3])))
{
}


/*static*/
BoundingBox2D BoundingBox2D::neutral_bbox()
{
    return BoundingBox2D(make_neutralBBox2D());
}

void BoundingBox2D::invalidate()
{
    this->bbox = make_neutralBBox2D();
}

bool BoundingBox2D::isNegligeable() const
{
    return minBBox().x() >= maxBBox().x() &&
           minBBox().y() >= maxBBox().y();
}

bool BoundingBox2D::isValid() const
{
    return minBBox().x() <= maxBBox().x() &&
           minBBox().y() <= maxBBox().y();
}

bool BoundingBox2D::isFlat() const
{
    return minBBox().x() == maxBBox().x() ||
           minBBox().y() == maxBBox().y();
}

bool BoundingBox2D::isNull() const
{
    return minBBox().x() == maxBBox().x() &&
           minBBox().y() == maxBBox().y();
}

BoundingBox2D::operator bbox_t() const
{
    return bbox;
}

SReal* BoundingBox2D::minBBoxPtr()
{
    return bbox.first.elems;
}

SReal* BoundingBox2D::maxBBoxPtr()
{
    return bbox.second.elems;
}

const SReal* BoundingBox2D::minBBoxPtr() const
{
    return bbox.first.elems;
}

const SReal* BoundingBox2D::maxBBoxPtr() const
{
    return bbox.second.elems;
}

const Vector2& BoundingBox2D::minBBox() const
{
    return bbox.first;
}

const Vector2& BoundingBox2D::maxBBox() const
{
    return bbox.second;
}

Vector2& BoundingBox2D::minBBox()
{
    return bbox.first;
}

Vector2& BoundingBox2D::maxBBox()
{
    return bbox.second;
}

bool BoundingBox2D::contains(const sofa::defaulttype::Vector2& point) const
{
    return  point.x() >= minBBox().x() && point.x() <= maxBBox().x() &&
            point.y() >= minBBox().y() && point.y() <= maxBBox().y();
}

bool BoundingBox2D::contains(const BoundingBox2D& other) const
{
    return contains(other.minBBox()) && contains(other.maxBBox());
}

bool BoundingBox2D::intersect(const BoundingBox2D& other) const
{
    if( other.minBBox().x() > maxBBox().x() || other.maxBBox().x ()< minBBox().x() ) return false;
    if( other.minBBox().y() > maxBBox().y() || other.maxBBox().y ()< minBBox().y() ) return false;
    return true;
}

void BoundingBox2D::intersection(const BoundingBox2D& other)
{
    minBBox().x() = std::max(minBBox().x(), other.minBBox().x());
    minBBox().y() = std::max(minBBox().y(), other.minBBox().y());

    maxBBox().x() = std::min(maxBBox().x(), other.maxBBox().x());
    maxBBox().y() = std::min(maxBBox().y(), other.maxBBox().y());
}

void BoundingBox2D::include(const sofa::defaulttype::Vector2& point)
{
    minBBox().x() = std::min( minBBox().x(), point.x());
    minBBox().y() = std::min( minBBox().y(), point.y());

    maxBBox().x() = std::max( maxBBox().x(), point.x());
    maxBBox().y() = std::max( maxBBox().y(), point.y());
}

void BoundingBox2D::include(const BoundingBox2D& other)
{
    minBBox().x() = std::min( minBBox().x(), other.minBBox().x());
    minBBox().y() = std::min( minBBox().y(), other.minBBox().y());

    maxBBox().x() = std::max( maxBBox().x(), other.maxBBox().x());
    maxBBox().y() = std::max( maxBBox().y(), other.maxBBox().y());
}

void BoundingBox2D::inflate(const SReal amount)
{
    Vector2 size(amount,amount);
    minBBox() -= size;
    maxBBox() += size;
}

BoundingBox2D BoundingBox2D::getIntersection( const BoundingBox2D& other ) const
{
    BoundingBox2D result;

    result.minBBox().x() = std::max(minBBox().x(), other.minBBox().x());
    result.minBBox().y() = std::max(minBBox().y(), other.minBBox().y());

    result.maxBBox().x() = std::min(maxBBox().x(), other.maxBBox().x());
    result.maxBBox().y() = std::min(maxBBox().y(), other.maxBBox().y());

    return result;
}

BoundingBox2D BoundingBox2D::getInclude( const sofa::defaulttype::Vector2& point ) const
{
    BoundingBox2D result;

    result.minBBox().x() = std::min( minBBox().x(), point.x());
    result.minBBox().y() = std::min( minBBox().y(), point.y());

    result.maxBBox().x() = std::max( maxBBox().x(), point.x());
    result.maxBBox().y() = std::max( maxBBox().y(), point.y());

    return result;
}

BoundingBox2D BoundingBox2D::getInclude( const BoundingBox2D& other ) const
{
    BoundingBox2D result;

    result.minBBox().x() = std::min( minBBox().x(), other.minBBox().x());
    result.minBBox().y() = std::min( minBBox().y(), other.minBBox().y());

    result.maxBBox().x() = std::max( maxBBox().x(), other.maxBBox().x());
    result.maxBBox().y() = std::max( maxBBox().y(), other.maxBBox().y());

    return result;
}

BoundingBox2D BoundingBox2D::getInflate( SReal amount ) const
{
    BoundingBox2D result;

    Vector2 size(amount,amount);
    result.minBBox() = minBBox() - size;
    result.maxBBox() = maxBBox() + size;

    return result;
}

//////////


namespace
{
BoundingBox1D::bbox_t make_neutralBBox1D()
{
    const SReal max_real = std::numeric_limits<SReal>::max();
    return std::make_pair(max_real,-max_real);
}
}

BoundingBox1D::BoundingBox1D()
    :bbox(make_neutralBBox1D())
{
}

BoundingBox1D::BoundingBox1D(const bbox_t& bbox)
    :bbox(bbox)
{
}

BoundingBox1D::BoundingBox1D(SReal minBBox, SReal maxBBox)
    :bbox(std::make_pair(minBBox,maxBBox))
{
}


BoundingBox1D::BoundingBox1D(const Vec2f& v )
    :bbox(std::make_pair((SReal)v[0],(SReal)v[1]))
{
}

BoundingBox1D::BoundingBox1D(const Vec2d& v )
     :bbox(std::make_pair((SReal)v[0],(SReal)v[1]))
{
}


/*static*/
BoundingBox1D BoundingBox1D::neutral_bbox()
{
    return BoundingBox1D(make_neutralBBox1D());
}

void BoundingBox1D::invalidate()
{
    this->bbox = make_neutralBBox1D();
}

bool BoundingBox1D::isNegligeable() const
{
    return minBBox() >= maxBBox();
}

bool BoundingBox1D::isValid() const
{
    return minBBox() <= maxBBox();
}

bool BoundingBox1D::isFlat() const
{
    return minBBox() == maxBBox();
}

bool BoundingBox1D::isNull() const
{
    return minBBox() == maxBBox();
}

BoundingBox1D::operator bbox_t() const
{
    return bbox;
}

const SReal& BoundingBox1D::minBBox() const
{
    return bbox.first;
}

const SReal& BoundingBox1D::maxBBox() const
{
    return bbox.second;
}

SReal& BoundingBox1D::minBBox()
{
    return bbox.first;
}

SReal& BoundingBox1D::maxBBox()
{
    return bbox.second;
}

bool BoundingBox1D::contains(SReal point) const
{
    return  point >= minBBox() && point <= maxBBox();
}

bool BoundingBox1D::contains(const BoundingBox1D& other) const
{
    return contains(other.minBBox()) && contains(other.maxBBox());
}

bool BoundingBox1D::intersect(const BoundingBox1D& other) const
{
    if( other.minBBox() > maxBBox() || other.maxBBox()< minBBox() ) return false;
    return true;
}

void BoundingBox1D::intersection(const BoundingBox1D& other)
{
    minBBox() = std::max(minBBox(), other.minBBox());
    maxBBox() = std::min(maxBBox(), other.maxBBox());
}

void BoundingBox1D::include(SReal point)
{
    minBBox() = std::min( minBBox(), point);
    maxBBox() = std::max( maxBBox(), point);
}

void BoundingBox1D::include(const BoundingBox1D& other)
{
    minBBox() = std::min( minBBox(), other.minBBox());
    maxBBox() = std::max( maxBBox(), other.maxBBox());
}

void BoundingBox1D::inflate(SReal amount)
{
    minBBox() -= amount;
    maxBBox() += amount;
}

BoundingBox1D BoundingBox1D::getIntersection( const BoundingBox1D& other ) const
{
    BoundingBox1D result;

    result.minBBox() = std::max(minBBox(), other.minBBox());
    result.maxBBox() = std::min(maxBBox(), other.maxBBox());

    return result;
}

BoundingBox1D BoundingBox1D::getInclude( SReal point ) const
{
    BoundingBox1D result;

    result.minBBox() = std::min( minBBox(), point );
    result.maxBBox() = std::max( maxBBox(), point );

    return result;
}

BoundingBox1D BoundingBox1D::getInclude( const BoundingBox1D& other ) const
{
    BoundingBox1D result;

    result.minBBox() = std::min( minBBox(), other.minBBox() );
    result.maxBBox() = std::max( maxBBox(), other.maxBBox() );

    return result;
}

BoundingBox1D BoundingBox1D::getInflate( SReal amount ) const
{
    BoundingBox1D result;

    result.minBBox() = minBBox() - amount;
    result.maxBBox() = maxBBox() + amount;

    return result;
}


} // namespace defaulttype

} // namespace sofa
