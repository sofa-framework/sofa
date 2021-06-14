#ifndef MESHDATAIMAGETOOLBOX_H
#define MESHDATAIMAGETOOLBOX_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/vector.h>


#include <string>

struct MeshDataImageToolBox
{
public:
    typedef sofa::type::Vec3d Coord3;
    typedef sofa::type::Vec3d Deriv3;
    typedef sofa::type::Vec<3, unsigned int> Index3;
    typedef sofa::type::Vec<4, unsigned int> Index4;
    typedef sofa::type::Vec<8, unsigned int> Index8;
    typedef sofa::type::vector< unsigned int > VecIndex;
    typedef sofa::type::vector< Index3 > VecIndex3;
    typedef sofa::type::vector< Index4 > VecIndex4;
    typedef sofa::type::vector< Index8 > VecIndex8;
    typedef sofa::type::vector< VecIndex > VecVecIndex;
    typedef sofa::type::vector< VecIndex3 > VecVecIndex3;
    typedef sofa::type::vector< VecIndex8 > VecVecIndex8;
    typedef sofa::type::vector< VecIndex4 > VecVecIndex4;

    typedef sofa::type::vector< Coord3 > VecCoord3;

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
    typedef sofa::type::vector< Layer > VecLayer;


    VecCoord3 positions;
    VecLayer veclayer;

    void generateFile()
    {

    }


};

#endif // MESHDATA_H
