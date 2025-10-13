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

#include <sofa/type/config.h>

#include <sofa/type/Vec.h>

#include <utility> /// for std::pair

namespace sofa::type
{

class SOFA_TYPE_API BoundingBox
{

public:
    typedef std::pair< sofa::type::Vec3, sofa::type::Vec3 > bbox_t;
    using Real = sofa::type::Vec3::value_type;

    constexpr BoundingBox()
        : BoundingBox(neutral_bbox().m_bbox)
    {}

    /// Define using the endpoints of the main diagonal
    constexpr BoundingBox(const sofa::type::Vec3& minBBox, const sofa::type::Vec3& maxBBox)
        : m_bbox({minBBox, maxBBox}) {}

    constexpr explicit BoundingBox(const bbox_t& bbox)
        : m_bbox(bbox)
    {}

    /// Define using xMin, xMax, yMin, yMax, zMin, zMax in this order
    constexpr BoundingBox(
        const Real xMin, const Real xMax,
        const Real yMin, const Real yMax,
        const Real zMin, const Real zMax )
        : BoundingBox({xMin, yMin, zMin}, {xMax, yMax, zMax})
    {}

    template<typename Scalar>
    constexpr explicit BoundingBox(const Vec<6, Scalar>& bbox)
        : BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5])
    {}

    static constexpr BoundingBox neutral_bbox()
    {
        constexpr Real max_real = std::numeric_limits<Real>::max();
        constexpr Real min_real = std::numeric_limits<Real>::lowest();
        return BoundingBox{
            {max_real, max_real, max_real},
            {min_real, min_real, min_real}
        };
    }

    operator bbox_t() const;

    [[nodiscard]] bool operator==(const BoundingBox& other) const;

    void invalidate();
    [[nodiscard]] bool isValid() const;
    [[nodiscard]] bool isFlat()  const;
    [[nodiscard]] bool isNegligible() const; // !valid || flat
    SOFA_ATTRIBUTE_DISABLED__BOUNDINGBOX_TYPO()
    [[nodiscard]] bool isNegligeable() const;
    [[nodiscard]] bool isNull()  const;

    SReal* minBBoxPtr();
    SReal* maxBBoxPtr();
    [[nodiscard]] const SReal* minBBoxPtr() const;
    [[nodiscard]] const SReal* maxBBoxPtr() const;
    [[nodiscard]] const sofa::type::Vec3&  minBBox() const;
    [[nodiscard]] const sofa::type::Vec3&  maxBBox() const;
    sofa::type::Vec3& minBBox();
    sofa::type::Vec3& maxBBox();

    [[nodiscard]] bool contains( const sofa::type::Vec3& point) const;
    [[nodiscard]] bool contains( const BoundingBox& other) const;

    [[nodiscard]] bool intersect( const BoundingBox& other) const;
    void intersection( const BoundingBox& other);

    void include( const sofa::type::Vec3& point);
    void include( const BoundingBox& other);

    void inflate( SReal amount );

    [[nodiscard]] BoundingBox getIntersection( const BoundingBox& other ) const;
    [[nodiscard]] BoundingBox getInclude( const sofa::type::Vec3& point ) const;
    [[nodiscard]] BoundingBox getInclude( const BoundingBox& other ) const;
    [[nodiscard]] BoundingBox getInflate( SReal amount ) const;

    friend std::ostream& operator << ( std::ostream& out, const BoundingBox& bbox)
    {
        out << bbox.minBBox() << " " <<  bbox.maxBBox();
        return out;
    }

    friend std::istream& operator >> ( std::istream& in, BoundingBox& bbox)
    {
        in >> bbox.minBBox() >> bbox.maxBBox();
        return in;
    }


protected:
    bbox_t m_bbox;
};


template <typename TReal>
class SOFA_ATTRIBUTE_DEPRECATED__TBOUNDINGBOX() TBoundingBox : public BoundingBox
{
public:
    TBoundingBox(const TReal* minBBoxPtr, const TReal* maxBBoxPtr)
        :BoundingBox(sofa::type::Vec3(minBBoxPtr),sofa::type::Vec3(maxBBoxPtr))
    {
    }

    TBoundingBox() : BoundingBox() {}
};



/// bounding rectangle
class SOFA_TYPE_API BoundingBox2D
{

public:
    typedef std::pair< sofa::type::Vec<2, SReal>, sofa::type::Vec<2, SReal> > bbox_t;

    BoundingBox2D();
    /// Define using the endpoints of the main diagonal
    BoundingBox2D(const sofa::type::Vec<2, SReal>& minBBox, const sofa::type::Vec<2, SReal>& maxBBox);
    BoundingBox2D(const bbox_t& bbox);
    /// Define using xmin, xmax, ymin, ymax in this order
    BoundingBox2D(SReal xmin, SReal xmax, SReal ymin, SReal ymax );
    /// Define using xmin, xmax, ymin, ymax in this order
    BoundingBox2D(const Vec4f& bbox);
    /// Define using xmin, xmax, ymin, ymax in this order
    BoundingBox2D(const Vec4d& bbox);

    static BoundingBox2D neutral_bbox();

    operator bbox_t() const;

    void invalidate();
    bool isValid() const;
    bool isFlat()  const;
    bool isNegligeable() const; // !valid || flat
    bool isNull()  const;

    SReal* minBBoxPtr();
    SReal* maxBBoxPtr();
    const SReal* minBBoxPtr() const;
    const SReal* maxBBoxPtr() const;
    const sofa::type::Vec<2, SReal>&  minBBox() const;
    const sofa::type::Vec<2, SReal>&  maxBBox() const;
    sofa::type::Vec<2, SReal>& minBBox();
    sofa::type::Vec<2, SReal>& maxBBox();

    bool contains( const sofa::type::Vec<2, SReal>& point) const;
    bool contains( const BoundingBox2D& other) const;

    bool intersect( const BoundingBox2D& other) const;
    void intersection( const BoundingBox2D& other);

    void include( const sofa::type::Vec<2, SReal>& point);
    void include( const BoundingBox2D& other);

    void inflate( SReal amount);

    BoundingBox2D getIntersection( const BoundingBox2D& other ) const;
    BoundingBox2D getInclude( const sofa::type::Vec<2, SReal>& point ) const;
    BoundingBox2D getInclude( const BoundingBox2D& other ) const;
    BoundingBox2D getInflate( SReal amount ) const;

    friend std::ostream& operator << ( std::ostream& out, const BoundingBox2D& bbox)
    {
        out << bbox.minBBox() << " " <<  bbox.maxBBox();
        return out;
    }

    friend std::istream& operator >> ( std::istream& in, BoundingBox2D& bbox)
    {
        in >> bbox.minBBox() >> bbox.maxBBox();
        return in;
    }


protected:
    bbox_t m_bbox;
};



/// bounding interval
class SOFA_TYPE_API BoundingBox1D
{

public:
    typedef std::pair< SReal, SReal > bbox_t;

    BoundingBox1D();
    /// Define using the endpoints of the main diagonal
    BoundingBox1D(SReal minBBox, SReal maxBBox);
    BoundingBox1D(const bbox_t& bbox);
    /// Define using xmin, xmax in this order
    BoundingBox1D(const Vec2f& bbox);
    /// Define using xmin, xmax in this order
    BoundingBox1D(const Vec2d& bbox);

    static BoundingBox1D neutral_bbox();

    operator bbox_t() const;

    void invalidate();
    bool isValid() const;
    bool isFlat()  const;
    bool isNegligeable() const; // !valid || flat
    bool isNull()  const;

    const SReal&  minBBox() const;
    const SReal&  maxBBox() const;
    SReal& minBBox();
    SReal& maxBBox();

    bool contains( SReal point) const;
    bool contains( const BoundingBox1D& other) const;

    bool intersect( const BoundingBox1D& other) const;
    void intersection( const BoundingBox1D& other);

    void include( SReal point);
    void include( const BoundingBox1D& other);

    void inflate( SReal amount);

    BoundingBox1D getIntersection( const BoundingBox1D& other ) const;
    BoundingBox1D getInclude( SReal point ) const;
    BoundingBox1D getInclude( const BoundingBox1D& other ) const;
    BoundingBox1D getInflate( SReal amount ) const;

    friend std::ostream& operator << ( std::ostream& out, const BoundingBox1D& bbox)
    {
        out << bbox.minBBox() << " " <<  bbox.maxBBox();
        return out;
    }

    friend std::istream& operator >> ( std::istream& in, BoundingBox1D& bbox)
    {
        in >> bbox.minBBox() >> bbox.maxBBox();
        return in;
    }


protected:
    bbox_t m_bbox;
};

} // namespace sofa::type
