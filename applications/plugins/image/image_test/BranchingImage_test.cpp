// #define BOOST_TEST_MODULE BranchingImage
 
#include <boost/test/unit_test.hpp>

#include "../BranchingImage.h"
#include <sofa/helper/RandomGenerator.h>


namespace fixture {

   using namespace cimg_library;


  template< class T >
  struct BranchingImageFixture {

      typedef sofa::defaulttype::Image<T> ImageTypes;
      typedef sofa::defaulttype::BranchingImage<T> BranchingImageTypes;

      sofa::helper::RandomGenerator randomGenerator;

      ImageTypes flatImage, flatImage2;
      BranchingImageTypes branchingImage, branchingImage2;

      BranchingImageFixture()
      {
          //// Prepare the flat image
          // set random dimensions
          typename ImageTypes::imCoord flatDim( unsigned(cimg::rand()*10+1), unsigned(cimg::rand()*10+1), unsigned(cimg::rand()*10+1), unsigned(cimg::rand()*4+1), unsigned(cimg::rand()*5+1) );

//          std::cerr<<flatDim<<std::endl;

          flatImage.setDimensions( flatDim );
          // initialize with random values
          cimglist_for(flatImage.getCImgList(),l)
                  cimg_forXYZC(flatImage.getCImg(l),x,y,z,c)
                  flatImage.getCImg(l)(x,y,z,c) = randomGenerator.random<T>( -1000, 1000 );

          // convert the flat image to a sparse branching image
          branchingImage = flatImage;

          // cloning
          branchingImage2 = branchingImage;

          branchingImage.toImage( flatImage2, 0 );
      }

      ~BranchingImageFixture()
      {
      }




      static bool dimensionsAreEqual( const typename ImageTypes::imCoord& flatDim, const typename BranchingImageTypes::Dimension& sparseDim )
      {
          return flatDim == sparseDim;
      }

      static int imagesAreEqual( const ImageTypes& flatImg, const BranchingImageTypes& sparseImg, bool valueTest = true, bool neighbourTest = false )
      {
          cimglist_for(flatImg.getCImgList(),l)
          {
                const CImg<T>& cimgl = flatImg.getCImg(l);
                const typename BranchingImageTypes::BranchingImage3D& iml = sparseImg.imgList[l];
                unsigned index1d = -1;
                cimg_forXYZ(cimgl,x,y,z)
                {
                    ++index1d;

                    //typename BranchingImageTypes::BranchingImageCIt imlxyz = iml.find(index1d/*sparseImg.index3Dto1D(x,y,z)*/);

//                    std::vector<const typename BranchingImageTypes::Element*> voxels;
//                    iml.find( index1d, voxels );

                    const typename BranchingImageTypes::SuperimposedVoxels& voxels = iml[index1d];

                    if( voxels.empty()/*iml[index1d].empty()*//*imlxyz == iml.end()*/ ) //the pixel x,y,z is not present in the branching image
                    {
                        if ( cimgl.get_vector_at(x,y,z).magnitude(1)!=0 ) return 1; // if the pixel is present in the flat image, there is a pb
                        else continue; // no pixel -> nothing to compare, test the next pixel
                    }

                    //const typename BranchingImageTypes::Voxels& voxels = imlxyz->second; // look at the superimposed voxels at position x,y,z
//                    const typename BranchingImageTypes::Voxels& voxels = iml[index1d];

                    if( voxels.size()>1 ) return 2; // the branching image has been built from a flat image, so there should be no superimposed voxels

                    if( valueTest )
                    {
                        for( unsigned c=0 ; c<flatImg.getDimensions()[3] ; ++c ) // for all channels
                            if( cimgl(x,y,z,c) != voxels[0][c] ) return 3; // check that the value is the same
                    }

                    if( neighbourTest )
                    {
                        // test neighbourhood connections
                        if( x>0 && ( ( cimgl.get_vector_at(x-1,y,z).magnitude(1)==0 ) == ( voxels[0].isNeighbour( typename BranchingImageTypes::VoxelIndex(index1d-1,0) ) ) ) ) return 4;
                        if( (unsigned)x<flatImg.getDimensions()[0]-1 && ( ( cimgl.get_vector_at(x+1,y,z).magnitude(1)==0 ) == ( voxels[0].isNeighbour( typename BranchingImageTypes::VoxelIndex(index1d+1,0) ) ) ) ) return 5;
                        if( y>0 && ( ( cimgl.get_vector_at(x,y-1,z).magnitude(1)==0 ) == ( voxels[0].isNeighbour( typename BranchingImageTypes::VoxelIndex(index1d-sparseImg.dimension[BranchingImageTypes::DIMENSION_X],0) ) ) ) ) return 6;
                        if( (unsigned)y<flatImg.getDimensions()[1]-1 && ( ( cimgl.get_vector_at(x,y+1,z).magnitude(1)==0 ) == ( voxels[0].isNeighbour( typename BranchingImageTypes::VoxelIndex(index1d+sparseImg.dimension[BranchingImageTypes::DIMENSION_X],0) ) ) ) ) return 7;
                        if( z>0 && ( ( cimgl.get_vector_at(x,y,z-1).magnitude(1)==0 ) == ( voxels[0].isNeighbour( typename BranchingImageTypes::VoxelIndex(index1d-sparseImg.sliceSize,0) ) ) ) ) return 8;
                        if( (unsigned)z<flatImg.getDimensions()[2]-1 && ( ( cimgl.get_vector_at(x,y,z+1).magnitude(1)==0 ) == ( voxels[0].isNeighbour( typename BranchingImageTypes::VoxelIndex(index1d+sparseImg.sliceSize,0) ) ) ) ) return 9;
                    }
                }
          }
          return 0;
      }


  };

}


typedef fixture::BranchingImageFixture<bool> BranchingImageFixtureB;
BOOST_FIXTURE_TEST_SUITE( BranchingImageB, BranchingImageFixtureB );
#include "BranchingImage_test.inl"
BOOST_AUTO_TEST_SUITE_END()

typedef fixture::BranchingImageFixture<double> BranchingImageFixtureD;
BOOST_FIXTURE_TEST_SUITE( BranchingImageD, BranchingImageFixtureD );
#include "BranchingImage_test.inl"
BOOST_AUTO_TEST_SUITE_END()
