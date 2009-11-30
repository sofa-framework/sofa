/*
 * TriangularConvexHull3D.h
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */

#ifndef CGALPLUGIN_TRIANGULARCONVEXHULL3D_H
#define CGALPLUGIN_TRIANGULARCONVEXHULL3D_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>

namespace cgal
{

template <class DataTypes>
class TriangularConvexHull3D : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularConvexHull3D,DataTypes),sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Tetra Tetra;

    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;


public:
    TriangularConvexHull3D();
    virtual ~TriangularConvexHull3D() { };

    void init();
    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const TriangularConvexHull3D<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Inputs
    Data<VecCoord> f_X0;

    //Outputs
    Data<VecCoord> f_newX0;
    Data<SeqTriangles> f_triangles;
};

#if defined(WIN32) && !defined(CGALPLUGIN_TRIANGULARCONVEXHULL3D_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API TriangularConvexHull3D<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API TriangularConvexHull3D<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} //cgal

#endif /* CGALPLUGIN_TriangularConvexHull3D_H */
