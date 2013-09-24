#ifndef MESHDATAIMAGETOOLBOX_H
#define MESHDATAIMAGETOOLBOX_H

#include <sofa/component/component.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/vector.h>




struct MeshDataImageToolBox
{
public:
    typedef sofa::defaulttype::Vec3d Coord3;
    typedef sofa::defaulttype::Vec3d Deriv3;
    typedef sofa::defaulttype::Vec<3, unsigned int> Index3;
    typedef sofa::defaulttype::Vec<4, unsigned int> Index4;
    typedef sofa::defaulttype::Vec<8, unsigned int> Index8;
    typedef sofa::helper::vector< unsigned int > VecIndex;
    typedef sofa::helper::vector< Index3 > VecIndex3;
    typedef sofa::helper::vector< Index4 > VecIndex4;
    typedef sofa::helper::vector< Index8 > VecIndex8;

    typedef sofa::helper::vector< Coord3 > VecCoord3;

    struct Layer
    {
        VecIndex elements;
        VecIndex4 grid1;
        VecIndex4 grid2;

        VecIndex3 triangles;
        VecIndex4 tetras;
        VecIndex8 hexas;

        void clear()
        {
            elements.clear();
            grid1.clear();
            grid2.clear();
            triangles.clear();
            tetras.clear();
            hexas.clear();
        }
    };
    typedef sofa::helper::vector< Layer > VecLayer;


    VecCoord3 positions;
    VecLayer veclayer;


};

#endif // MESHDATA_H
