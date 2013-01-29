// #define BOOST_TEST_MODULE SparseImage
 
#include <boost/test/unit_test.hpp>

#include "../SparseImage.h"
#include <sofa/helper/RandomGenerator.h>


namespace fixture {

   using namespace cimg_library;


  template< class T >
  struct SparseImageFixture {

      typedef sofa::defaulttype::Image<T> FlatImage;
      typedef sofa::defaulttype::SparseImage<T> SparseBranchingImage;

      sofa::helper::RandomGenerator randomGenerator;

      FlatImage flatImage;
      SparseBranchingImage sparseImage;

      SparseImageFixture()
      {
          //// Prepare the flat image
          // set random dimensions
          typename FlatImage::imCoord flatDim( unsigned(cimg::rand()*10+1), unsigned(cimg::rand()*10+1), unsigned(cimg::rand()*10+1), unsigned(cimg::rand()*4+1), unsigned(cimg::rand()*5+1) );

//          std::cerr<<flatDim<<std::endl;

          flatImage.setDimensions( flatDim );
          // initialize with random values
          cimglist_for(flatImage.getCImgList(),l)
                  cimg_forXYZC(flatImage.getCImg(l),x,y,z,c)
                  flatImage.getCImg(l)(x,y,z,c) = randomGenerator.randomReal<T>();

          // convert the flat image to a sparse branching image
          sparseImage = flatImage;
      }

      ~SparseImageFixture()
      {
      }




      static bool dimensionsAreEqual( const typename FlatImage::imCoord& flatDim, const typename SparseBranchingImage::Dimension& sparseDim )
      {
          return flatDim == sparseDim;
      }

      // should return true if sparseImg has been built from flatImg
      static bool imagesAreEqual( const FlatImage& flatImg, const SparseBranchingImage& sparseImg )
      {
          cimglist_for(flatImg.getCImgList(),l)
          {
                const CImg<T>& cimgl = flatImg.getCImg(l);
                const typename SparseBranchingImage::BranchingImage& iml = sparseImg.imgList[l];
                unsigned index1d = -1;
                cimg_forXYZ(cimgl,x,y,z)
                {
                    ++index1d;

                    typename SparseBranchingImage::BranchingImageCIt imlxyz = iml.find(index1d/*sparseImg.index3Dto1D(x,y,z)*/);

                    if( imlxyz == iml.end() ) //the pixel x,y,z is not present in the branching image
                    {
                        if ( cimgl.get_vector_at(x,y,z).magnitude(1)!=0 ) return false; // if the pixel is present in the flat image, there is a pb
                        else continue; // no pixel -> nothing to compare, test the next pixel
                    }

                    const typename SparseBranchingImage::Voxels& voxels = imlxyz->second; // look at the superimposed voxels at position x,y,z

                    if( voxels.size()>1 ) return false; // the branching image has been built from a flat image, so there should be no superimposed voxels

                    for( unsigned c=0 ; c<flatImg.getDimensions()[3] ; ++c ) // for all channels
                        if( cimgl(x,y,z,c) != voxels[0][c] ) return false; // check that the value is the same

                    // test neighbourood connections
                    if( x>0 && ( ( cimgl.get_vector_at(x-1,y,z).magnitude(1)==0 ) != voxels[0].connections[sofa::defaulttype::SparseImageB::Voxel::LEFT].empty() ) ) return false;
                    if( (unsigned)x<flatImg.getDimensions()[0]-1 && ( ( cimgl.get_vector_at(x+1,y,z).magnitude(1)==0 ) != voxels[0].connections[sofa::defaulttype::SparseImageB::Voxel::RIGHT].empty() ) ) return false;
                    if( y>0 && ( ( cimgl.get_vector_at(x,y-1,z).magnitude(1)==0 ) != voxels[0].connections[sofa::defaulttype::SparseImageB::Voxel::BOTTOM].empty() ) ) return false;
                    if( (unsigned)y<flatImg.getDimensions()[1]-1 && ( ( cimgl.get_vector_at(x,y+1,z).magnitude(1)==0 ) != voxels[0].connections[sofa::defaulttype::SparseImageB::Voxel::TOP].empty() ) ) return false;
                    if( z>0 && ( ( cimgl.get_vector_at(x,y,z-1).magnitude(1)==0 ) != voxels[0].connections[sofa::defaulttype::SparseImageB::Voxel::BACK].empty() ) ) return false;
                    if( (unsigned)z<flatImg.getDimensions()[2]-1 && ( ( cimgl.get_vector_at(x,y,z+1).magnitude(1)==0 ) != voxels[0].connections[sofa::defaulttype::SparseImageB::Voxel::FRONT].empty() ) ) return false;
                }
          }
          return true;
      }


  };

}


typedef fixture::SparseImageFixture<bool> SparseImageFixtureB;
BOOST_FIXTURE_TEST_SUITE( SparseImageB, SparseImageFixtureB );
#include "SparseImage_test.inl"
BOOST_AUTO_TEST_SUITE_END()

typedef fixture::SparseImageFixture<double> SparseImageFixtureD;
BOOST_FIXTURE_TEST_SUITE( SparseImageD, SparseImageFixtureD );
#include "SparseImage_test.inl"
BOOST_AUTO_TEST_SUITE_END()
