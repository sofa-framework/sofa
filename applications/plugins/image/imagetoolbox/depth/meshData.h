#ifndef MESHDATAIMAGETOOLBOX_H
#define MESHDATAIMAGETOOLBOX_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/vector.h>


#include <string>

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
    typedef sofa::helper::vector< VecIndex > VecVecIndex;
    typedef sofa::helper::vector< VecIndex3 > VecVecIndex3;
    typedef sofa::helper::vector< VecIndex8 > VecVecIndex8;
    typedef sofa::helper::vector< VecIndex4 > VecVecIndex4;

    typedef sofa::helper::vector< Coord3 > VecCoord3;

    struct Layer
    {
        VecIndex elements;
        VecIndex4 grid1;
        VecIndex4 grid2;

        VecIndex3 triangles;
        VecIndex4 tetras;
        VecIndex8 hexas;

        std::string name;

        unsigned int nbSlice;
        VecVecIndex positionIndexOfSlice;
        VecVecIndex edgePositionIndexOfSlice;
        VecVecIndex hexaIndexOfSlice;
        VecVecIndex tetraIndexOfSlice;


        void clear()
        {
            nbSlice = 0;
            elements.clear();
            grid1.clear();
            grid2.clear();
            triangles.clear();
            tetras.clear();
            hexas.clear();
            positionIndexOfSlice.clear();
            edgePositionIndexOfSlice.clear();
            hexaIndexOfSlice.clear();
            tetraIndexOfSlice.clear();
        }

        void setSlice(unsigned int slice)
        {
            nbSlice = slice;
            positionIndexOfSlice.resize(slice+1);
            edgePositionIndexOfSlice.resize(slice+1);
            hexaIndexOfSlice.resize(slice);
            tetraIndexOfSlice.resize(slice);
        }

    };
    typedef sofa::helper::vector< Layer > VecLayer;


    VecCoord3 positions;
    VecLayer veclayer;

    void generateFile()
    {

    }


};

#endif // MESHDATA_H
