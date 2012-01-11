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
#ifndef SOFA_DEFAULTTYPE_BOUNDINGBOX_H
#define SOFA_DEFAULTTYPE_BOUNDINGBOX_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/defaulttype.h>


namespace sofa
{
namespace defaulttype
{


class SOFA_DEFAULTTYPE_API BoundingBox
{

public:
    typedef std::pair< Vector3, Vector3 > bbox_t;

    BoundingBox();
    /// Define using the endpoints of the main diagonal
    BoundingBox(const Vector3& minBBox, const Vector3& maxBBox);
    BoundingBox(const bbox_t& bbox);
    /// Define using xmin, xmax, ymin, ymax, zmin, zmax in this order
    BoundingBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax );
    /// Define using xmin, xmax, ymin, ymax, zmin, zmax in this order
    BoundingBox(const Vec6f& bbox);
    /// Define using xmin, xmax, ymin, ymax, zmin, zmax in this order
    BoundingBox(const Vec6d& bbox);

    static BoundingBox neutral_bbox();

    operator bbox_t() const;

    void invalidate();
    bool isValid() const;
    bool isFlat()  const;

    SReal* minBBoxPtr();
    SReal* maxBBoxPtr();
    const SReal* minBBoxPtr() const;
    const SReal* maxBBoxPtr() const;
    const Vector3&  minBBox() const;
    const Vector3&  maxBBox() const;
    Vector3& minBBox();
    Vector3& maxBBox();





    bool contains( const sofa::defaulttype::Vector3& point) const;
    bool contains( const BoundingBox& other) const;

    bool intersect( const BoundingBox& other) const;
    void intersection( const BoundingBox& other);

    void include( const sofa::defaulttype::Vector3& point);
    void include( const BoundingBox& other);

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
    bbox_t bbox;
};


template <typename TReal>
class TBoundingBox : public BoundingBox
{
public:
    TBoundingBox(const TReal* minBBoxPtr, const TReal* maxBBoxPtr)
        :BoundingBox(Vector3(minBBoxPtr),Vector3(maxBBoxPtr))
    {
    }

};

}
}

#endif // SOFA_DEFAULTTYPE_BOUNDINGBOX_H
