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
    BoundingBox(const Vector3& minBBox,
            const Vector3& maxBBox);
    BoundingBox(const bbox_t& bbox);

    static BoundingBox neutral_bbox();

    operator bbox_t() const;

    void invalidate();

    SReal* minBBoxPtr();
    SReal* maxBBoxPtr();
    const Vector3&  minBBox() const;
    const Vector3&  maxBBox() const;
    Vector3& minBBox();
    Vector3& maxBBox();





    bool contains( const sofa::defaulttype::Vector3& point);
    bool contains( const BoundingBox& other);

    bool intersect( const BoundingBox& other);
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
    };

};

}
}

#endif // SOFA_DEFAULTTYPE_BOUNDINGBOX_H
