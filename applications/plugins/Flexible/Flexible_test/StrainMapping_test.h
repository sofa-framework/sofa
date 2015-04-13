#ifndef StrainMapping_Test_H__
#define StrainMapping_Test_H__



#include "stdafx.h"
#include <sofa/helper/Quater.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"


#include <Mapping_test.h>


namespace sofa {

    using std::cout;
    using std::cerr;
    using std::endl;
    using namespace core;
    using namespace component;
    using defaulttype::Vec;
    using defaulttype::Mat;
    using testing::Types;

    using namespace component::mapping;


    /// Base class to compare StrainMapping
    template <typename _Mapping>
    struct StrainMappingTest : public Mapping_test<_Mapping>
    {
        typedef Mapping_test<_Mapping> Inherited;

        typedef typename Inherited::In In;
        typedef typename Inherited::WriteInVecCoord WriteInVecCoord;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename Inherited::InVecCoord InVecCoord;
        typedef typename In::Frame InFrame;

        bool runTest( defaulttype::Mat<3,3,Real>& rotation, defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real>& strain, const OutVecCoord& expectedChildCoords)
        {
            InVecCoord xin(1);
            OutVecCoord xout(1);

            // parent position
            InFrame &f = xin[0].getF();

            // stretch + shear
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
            {
                for( unsigned int j=0 ; j<In::material_dimensions ; ++j )
                {
                    f[i][j] = strain[i][j];
                }
            }

            // rotation
            f = rotation * f;
//            cerr<<"StrainMappingTest::runTest, f="<< f << endl;
//            cerr<<"StrainMappingTest::runTest, expected="<< expectedChildCoords << endl;

            static_cast<_Mapping*>(this->mapping)->assemble.setValue(true);

            return Inherited::runTest(xin,xout,xin,expectedChildCoords);
        }

    };


} // namespace sofa

#endif
