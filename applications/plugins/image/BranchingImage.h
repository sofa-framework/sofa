
#ifndef IMAGE_BranchingImage_H
#define IMAGE_BranchingImage_H



#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <map>

#include "ImageTypes.h"
#include "Containers.h"

namespace sofa
{

namespace defaulttype
{

using helper::vector;
using helper::NoPreallocationVector;




/// A BranchingImage is an array (size of t) of vectors (one vector per pixel (x,y,z)).
/// Each pixel corresponds to a SuperimposedVoxels, alias a vector of ConnectionVoxel.
/// a ConnectionVoxel stores a value for each channels + its neighbours indices
/// Nesme, Kry, Jeřábková, Faure, "Preserving Topology and Elasticity for Embedded Deformable Models", Siggraph09
template<typename _T>
struct BranchingImage
{

public:

    /// stored type
    typedef _T T;

    /// each direction around a voxel
    typedef enum { BACK=0, Zm1=BACK, BOTTOM=1, Ym1=BOTTOM, LEFT=2, Xm1=LEFT, FRONT=3, Zp1=FRONT, TOP=4, Yp1=TOP, RIGHT=5, Xp1=RIGHT, NB_NeighbourDirections=6 } NeighbourDirection;
    /// returns the opposite direction of a given direction  left->right,  right->left
    inline NeighbourDirection oppositeDirection( NeighbourDirection d ) const { return NeighbourDirection( (d+3)%NB_NeighbourDirections ); }

    /// a ConnectionVoxel stores a value for each channels + its neighbour indices /*+ its 1D index in the image*/
    /// NB: a ConnectionVoxel does not know its spectrum size (nb of channels) to save memory
    class ConnectionVoxel
    {

    public:

        /// default constructor = no allocation
        ConnectionVoxel() : value(0) {}
        /// with allocation constructor
        ConnectionVoxel( size_t size ) { value = new T[size]; }
        ~ConnectionVoxel() { if( value ) delete [] value; }

        /// copy
        /// @warning neighbourood is copied but impossible to check its validity
        void clone( const ConnectionVoxel& cv, unsigned spectrum )
        {
            if( value ) delete [] value;
            value = new T[spectrum];
            memcpy( value, cv.value, spectrum*sizeof(T) );
            //index = cv.index;
            neighbours = cv.neighbours;
        }

        /// alloc or realloc without keeping existing data and without initialization
        void resize( size_t newSize )
        {
            if( !value ) value = new T[newSize];
            else { delete [] value; value = new T[newSize]; } // could do a realloc keeping existing data
        }

        /// computes a norm over all channels
        double magnitude( unsigned spectrum, const int magnitude_type=2 ) const
        {
            double res = 0;
            switch (magnitude_type) {
            case -1 : {
                for( unsigned i=0 ; i<spectrum ; ++i ) { const double val = (double)abs(value[i]); if (val>res) res = val; }
            } break;
            case 1 : {
              for( unsigned i=0 ; i<spectrum ; ++i ) res += (double)abs(value[i]);
            } break;
            default : {
              for( unsigned i=0 ; i<spectrum ; ++i ) res += (double)(value[i]*value[i]);
              res = (double)sqrt(res);
            }
            }
            return res;
        }

        /// @returns the min channel value
        T min( unsigned spectrum ) const
        {
            if( !value ) return 0;
            T m = value[0];
            for( unsigned i=1 ; i<spectrum ; ++i ) if( value[i]<m ) m=value[i];
            return m;
        }

        /// @returns the max channel value
        T max( unsigned spectrum ) const
        {
            if( !value ) return 0;
            T m = value[0];
            for( unsigned i=1 ; i<spectrum ; ++i ) if( value[i]>m ) m=value[i];
            return m;
        }

        /// @returns true iff all channels are 0
        bool empty( unsigned spectrum ) const
        {
            for( unsigned i=1 ; i<spectrum ; ++i ) if( value[i] ) return false;
            return true;
        }

        //unsigned index; ///< the 1D position in the BranchingImage3D // @TODO is it really necessary

        T* value; ///< value of the voxel for each channel (value is the size of the C dimension of the ConnectionImage)

        typedef NoPreallocationVector<unsigned> NeighboursInOneDirection;
        typedef Vec< NB_NeighbourDirections, NeighboursInOneDirection > Neighbours;
        Neighbours neighbours; ///< neighbours of the voxels. In each 6 directions (bottom, up, left...), a list of all connected voxels (indices in the Voxels list of the neighbour pixel in the ConnectionImage)



        /// accessor
        /// @warning index must be less than the spectrum
        T& operator [] (size_t index) const
        {
            return value[ index ];
        }

        /// equivalent to ==
        bool isEqual( const ConnectionVoxel& other, unsigned spectrum ) const
        {
            for( unsigned i=0 ; i<spectrum ; ++i )
                if( value[i] != other.value[i] ) return false;
            return true;
        }

        /// add the given voxel as a neigbour
        /// @warning it is doing only one way (this has to be added as a neighbour of n)
        /// if testUnicity==true, the neighbour is added only if it is not already there
        void addNeighbour( NeighbourDirection d, unsigned neighbourOffset, bool testUnicity = false )
        {
            if( !testUnicity || !isNeighbour(d,neighbourOffset) ) neighbours[d].push_back( neighbourOffset );
        }

        /// is the given voxel a neighbour of this voxel?
        bool isNeighbour( NeighbourDirection d, unsigned neighbourOffset ) const
        {
            return neighbours[d].find( neighbourOffset ) != -1;
        }


    private:

        // no pure copy constructor (spectrum is needed to be able to copy the ConnectionVoxel)
        ConnectionVoxel( const ConnectionVoxel& ) { assert(false); }
        void operator=( const ConnectionVoxel& ) { assert(false); }
        bool operator==( const ConnectionVoxel& ) const { assert(false); }

    }; // class ConnectionVoxel




    /// An array of ConnectionVoxel
    class SuperimposedVoxels : public NoPreallocationVector< ConnectionVoxel >
    {

    public:

        typedef NoPreallocationVector< ConnectionVoxel > Inherited;

        SuperimposedVoxels() : Inherited() {}

        /// add a superimposed voxel
        void push_back( const ConnectionVoxel& v, unsigned spectrum )
        {
            Inherited::resizeAndKeep( this->_size+1 );
            this->last().clone( v, spectrum );
        }

        /// copy superimposed voxels
        /// @warning about voxel connectivity
        void clone( const SuperimposedVoxels& other, unsigned spectrum )
        {
            resize( other._size );
            for( unsigned i=0 ; i<this->_size ; ++i )
            {
                this->_array[i].clone( other._array[i], spectrum );
            }
        }

        /// equivalent to ==
        bool isEqual( const SuperimposedVoxels& other, unsigned spectrum ) const
        {
            if( this->_size != other._size ) return false;
            for( unsigned i=0 ; i<this->_size ; ++i )
                if( !this->_array[i].isEqual( other._array[i], spectrum ) ) return false;
            return true;
        }

        /// convert to a unique voxel
        /// conversionType : 0->first voxel, 1->average
        void toFlatVoxel( T& v, unsigned conversionType, unsigned channel ) const
        {
            if( this->empty() ) return;

            switch( conversionType )
            {
            case 1:
                v = this->_array[0][channel];
                for( unsigned i=1 ; i<this->_size ; ++i )
                    v += this->_array[i][channel];
                v = (T)( v / (float)this->_size );
                break;
            case 0:
            default:
                v = this->_array[0][channel];
                break;
            }
        }


        /// all needed operators +, +=, etc. can be overloaded here


    private :

        /// impossible to copy a ConnectedVoxel without the spectrum size
        void push_back( const ConnectionVoxel& v ) { assert(false); }
        SuperimposedVoxels( const SuperimposedVoxels& cv ) { assert(false); }
        void operator=( const SuperimposedVoxels& ) { assert(false); }
        bool operator==( const SuperimposedVoxels& ) const { assert(false); }

    }; // class SuperimposedVoxels



    /// a BranchingImage is a dense image with a vector of SuperimposedVoxels at each pixel
    class BranchingImage3D : public NoPreallocationVector<SuperimposedVoxels>
    {
     public:

        typedef NoPreallocationVector<SuperimposedVoxels> Inherited;

        BranchingImage3D() : Inherited() {}

        /// copy
        void clone( const BranchingImage3D& other, unsigned spectrum )
        {
            resize( other._size );
            for( unsigned i=0 ; i<this->_size ; ++i )
            {
                this->_array[i].clone( other._array[i], spectrum );
            }
        }

        /// equivalent to ==
        bool isEqual( const BranchingImage3D& other, unsigned spectrum ) const
        {
            if( this->_size != other._size ) return false;
            for( unsigned i=0 ; i<this->_size ; ++i )
                if( !this->_array[i].isEqual( other._array[i], spectrum ) ) return false;
            return true;
        }

        /// \returns the offset of the given ConnectionVoxel in its SuperImposedVoxels vector
        int getOffset( unsigned index1d, const ConnectionVoxel& v ) const
        {
            return this->_array[index1d].getOffset( &v );
        }

    private:

        /// impossible to copy a ConnectedVoxel without the spectrum size
        BranchingImage3D( const BranchingImage3D& ) { assert(false); }
        void operator=( const BranchingImage3D& ) { assert(false); }
        bool operator==( const BranchingImage3D& ) const { assert(false); }
        void push_back( const SuperimposedVoxels& v ) { assert(false); }

    }; // class BranchingImage



    /// the 5 dimension labels of an image ( x, y, z, spectrum=nb channels , time )
    typedef enum{ DIMENSION_X=0, DIMENSION_Y, DIMENSION_Z, DIMENSION_S /* spectrum = nb channels*/, DIMENSION_T /*4th dimension = time*/, NB_DimensionLabel } DimensionLabel;
    /// the 5 dimensions of an image ( x, y, z, spectrum=nb channels , time )
    typedef Vec<NB_DimensionLabel,unsigned int> Dimension; // [x,y,z,s,t]


    Dimension dimension; ///< the image dimensions [x,y,z,s,t]
    unsigned sliceSize; ///< (x,y) slice size
    unsigned imageSize; ///< (x,y,z) image size
    BranchingImage3D* imgList; ///< array of BranchingImage over time t



    static const char* Name();

    ///constructors/destructors
    BranchingImage() : dimension(), imgList(0) {}
    ~BranchingImage()
    {
        if( imgList ) delete [] imgList;
    }


    /// copy constructor
    BranchingImage(const BranchingImage<T>& img) : dimension(), imgList(0)
    {
        *this = img;
    }

    /// clone
    BranchingImage<T>& operator=(const BranchingImage<T>& im)
    {
        // allocate & copy everything
        setDimension( im.getDimension() );

        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            imgList[t].clone( im.imgList[t], dimension[DIMENSION_S] );
        }

        return *this;
    }


    /// conversion from flat image to connection image
    BranchingImage(const Image<T>& img)
    {
        *this = img;
    }

    /// conversion from flat image to connection image
    BranchingImage<T>& operator=(const Image<T>& im)
    {
        setDimension( im.getDimensions() );

        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            BranchingImage3D& imt = imgList[t];
            const CImg<T>& cimgt = im.getCImg(t);
            unsigned index1D = 0;
            cimg_forXYZ(cimgt,x,y,z)
            {
                CImg<long double> vect=cimgt.get_vector_at(x,y,z);
                if( vect.magnitude(1) != 0 )
                {
//                    assert( index1D == index3Dto1D(x,y,z) );
                    {
                        ConnectionVoxel v( dimension[DIMENSION_S] );
                        for( unsigned c = 0 ; c<dimension[DIMENSION_S] ; ++c )
                            v[c] = cimgt(x,y,z,c);
//                        v.index = index1D;
                        v.neighbours.clear();
                        imt[index1D].push_back( v, dimension[DIMENSION_S] );
                    }
                    // neighbours
                    if( x>0 && !imt[index1D-1].empty() ) { imt[index1D][0].addNeighbour( LEFT, 0 ); imt[index1D-1][0].addNeighbour( RIGHT, 0 ); }
                    if( y>0 && !imt[index1D-dimension[DIMENSION_X]].empty() ) { imt[index1D][0].addNeighbour( BOTTOM, 0 ); imt[index1D-dimension[DIMENSION_X]][0].addNeighbour( TOP, 0 ); }
                    if( z>0 && !imt[index1D-sliceSize].empty() ) { imt[index1D][0].addNeighbour( BACK, 0 ); imt[index1D-sliceSize][0].addNeighbour( FRONT, 0 ); }
                }
                ++index1D;
            }
        }

        return *this;
    }


    /// conversion to a flat image
    /// conversionType : 0->first voxel, 1->average
    void toImage( Image<T>& img, unsigned conversionType = 0 ) const
    {
        img.clear();
        typename Image<T>::imCoord dim = dimension;
        img.setDimensions( dim );
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            const BranchingImage3D& bimt = imgList[t];
            CImg<T>& cimgt = img.getCImg(t);

            cimgt.fill((T)0);

            unsigned index1D = 0;
            cimg_forXYZ(cimgt,x,y,z)
            {
                cimg_forC( cimgt, c )
                    bimt[index1D].toFlatVoxel( cimgt(x,y,z,c), conversionType, c );

                ++index1D;
            }
        }
    }


    /// delete everything, free memory
    void clear()
    {
        if( imgList )
        {
            delete [] imgList;
            imgList = 0;
        }
    }



    /// compute the map key in BranchingImage from the pixel position
    inline unsigned index3Dto1D( unsigned x, unsigned y, unsigned z ) const
    {
        return ( z * dimension[DIMENSION_Y]  + y ) * dimension[DIMENSION_X] + x;
    }

    /// compute the pixel position from the map key in BranchingImage
    inline void index1Dto3D( unsigned key, unsigned& x, unsigned& y, unsigned& z ) const
    {
//        x = key % dimension[DIMENSION_X];
//        y = ( key / dimension[DIMENSION_X] ) % dimension[DIMENSION_Y];
//        z = key / sliceSize;
        y = key / dimension[DIMENSION_X];
        x = key - y * dimension[DIMENSION_X];
        z = y / dimension[DIMENSION_Y];
        y = y - z * dimension[DIMENSION_Y];
    }

    /// \returns the index of the neighbour in the given direction (index in the BranchingImage3D)
    inline unsigned getNeighbourIndex( NeighbourDirection d, unsigned voxelIndex ) const
    {
        switch(d)
        {
            case LEFT: return voxelIndex-1; break;
            case RIGHT: return voxelIndex+1; break;
            case BOTTOM: return voxelIndex-dimension[DIMENSION_X]; break;
            case TOP: return voxelIndex+dimension[DIMENSION_X]; break;
            case BACK: return voxelIndex-sliceSize; break;
            case FRONT: return voxelIndex+sliceSize; break;
            default: return 0;
        }
    }

    /// \returns the direction between two neighbour voxels
    /// @warnings the two given voxels are supposed to be neighbours, otherwise NB_NeighbourDirections is returned
    /// example: returning LEFT means neighbourIndex is at the LEFT position of index
    inline NeighbourDirection getDirection( unsigned index, unsigned neighbourIndex ) const
    {
        int offset = neighbourIndex - index;
        if( offset==-1 ) return LEFT;
        else if( offset==1 ) return RIGHT;
        else if( offset==-dimension[DIMENSION_X] ) return BOTTOM;
        else if( offset==dimension[DIMENSION_X] ) return TOP;
        else if( offset==-sliceSize ) return BACK;
        else if( offset==sliceSize ) return FRONT;
        else return NB_NeighbourDirections;
    }



    /// \returns the 5 image dimensions (x,y,z,s,t)
    const Dimension& getDimension() const
    {
        return dimension;
    }

    /// resizing
    /// @warning data is deleted
    void setDimension( const Dimension& newDimension )
    {
        clear();

        for( unsigned i=0 ; i<NB_DimensionLabel ; ++i ) if( !newDimension[i] ) { dimension.clear(); imageSize = sliceSize = 0; return; }

        dimension = newDimension;
        sliceSize = dimension[DIMENSION_X] * dimension[DIMENSION_Y];
        imageSize = sliceSize * dimension[DIMENSION_Z];
        imgList = new BranchingImage3D[dimension[DIMENSION_T]];
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
            imgList[t].resize( imageSize );
    }

    /// write dimensions
    inline friend std::istream& operator >> ( std::istream& in, BranchingImage<T>& im )
    {
        Dimension dim;
        in >> dim;
        im.setDimension( dim );
        return in;
    }

    /// read dimensions
    friend std::ostream& operator << ( std::ostream& out, const BranchingImage<T>& im )
    {
        out << im.getDimension();
        return out;
    }

    /// comparison
    bool operator==( const BranchingImage<T>& other ) const
    {
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
            if( !imgList[t].isEqual( other.imgList[t], dimension[DIMENSION_S] ) ) return false;
        return true;
    }

    /// comparison
    bool operator!=( const BranchingImage<T>& other ) const
    {
        return !(*this==other);
    }



    /// \returns an approximative size in bytes, useful for debugging
    size_t approximativeSizeInBytes() const
    {
        size_t total = dimension[DIMENSION_T]*(imageSize+1)*( sizeof(unsigned) + sizeof(void*) ); // superimposed voxel vectors + BranchingImage3D vector

        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t ) // per BranchingImage3D
        {
            const BranchingImage3D& imt = imgList[t];
            for( unsigned int index=0 ; index<imt.size() ; ++index ) // per SumperimposedVoxels
            {
                total += imt[index].size() * ( /*sizeof(unsigned)*/ /*index*/ +
                                               sizeof(void*) /* channel vector*/ +
                                               dimension[DIMENSION_S]*sizeof(T) /*channel entries*/ +
                                               NB_NeighbourDirections * ( sizeof(unsigned) + sizeof(void*) ) /* 6 neighbour vectors per voxel*/
                                               );

                for( unsigned v=0 ; v<imt[index].size() ; ++v ) // per ConnnectedVoxel
                {
                    for( unsigned d=0 ; d<NB_NeighbourDirections ; ++d )
                    {
                        total += imt[index][v].neighbours[d].size() * sizeof( T* ); // neighbour entries
                    }
                }
            }
        }
        return total;
    }

    /// check neighbourood validity
    int isNeighbouroodValid() const
    {
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            const BranchingImage3D& imt = imgList[t];
            unsigned index1d = 0;
            for( unsigned z = 0 ; z < dimension[DIMENSION_Z] ; ++z )
            {
                for( unsigned y = 0 ; y < dimension[DIMENSION_Y] ; ++y )
                {
                    for( unsigned x = 0 ; x < dimension[DIMENSION_X] ; ++x )
                    {
                        const SuperimposedVoxels& voxels = imt[index1d];
                        for( unsigned v = 0 ; v < voxels.size() ; ++v )
                        {
                            const typename ConnectionVoxel::Neighbours& neighbours = voxels[v].neighbours;
                            for( unsigned d = 0 ; d < NB_NeighbourDirections ; ++d )
                            {
                                const typename ConnectionVoxel::NeighboursInOneDirection& neighboursOneDirection = neighbours[d];
                                for( unsigned n = 0 ; n < neighboursOneDirection.size() ; ++n )
                                {
                                    unsigned neighbourIndex = getNeighbourIndex( (NeighbourDirection)d, index1d );
                                    if( neighboursOneDirection[n] >= imt[neighbourIndex].size() )  return 1; // there is nobody where there should be the neighbour
                                    if( !imt[neighbourIndex][neighboursOneDirection[n]].isNeighbour( oppositeDirection((NeighbourDirection)d), imt.getOffset(index1d,voxels[v]) ) ) return 2; // complementary neighbour is no inserted
//                                    if( imt[neighbourIndex][neighboursOneDirection[n]].index != neighbourIndex || voxels[v].index != index1d ) return false; // a voxel has a good index
                                }
                            }
                        }
                        ++ index1d;
                    }
                }
            }
        }
        return 0;
    }


};


typedef BranchingImage<char> BranchingImageC;
typedef BranchingImage<unsigned char> BranchingImageUC;
typedef BranchingImage<int> BranchingImageI;
typedef BranchingImage<unsigned int> BranchingImageUI;
typedef BranchingImage<short> BranchingImageS;
typedef BranchingImage<unsigned short> BranchingImageUS;
typedef BranchingImage<long> BranchingImageL;
typedef BranchingImage<unsigned long> BranchingImageUL;
typedef BranchingImage<float> BranchingImageF;
typedef BranchingImage<double> BranchingImageD;
typedef BranchingImage<bool> BranchingImageB;

template<> inline const char* BranchingImageC::Name() { return "BranchingImageC"; }
template<> inline const char* BranchingImageUC::Name() { return "BranchingImageUC"; }
template<> inline const char* BranchingImageI::Name() { return "BranchingImageI"; }
template<> inline const char* BranchingImageUI::Name() { return "BranchingImageUI"; }
template<> inline const char* BranchingImageS::Name() { return "BranchingImageS"; }
template<> inline const char* BranchingImageUS::Name() { return "BranchingImageUS"; }
template<> inline const char* BranchingImageL::Name() { return "BranchingImageL"; }
template<> inline const char* BranchingImageUL::Name() { return "BranchingImageUL"; }
template<> inline const char* BranchingImageF::Name() { return "BranchingImageF"; }
template<> inline const char* BranchingImageD::Name() { return "BranchingImageD"; }
template<> inline const char* BranchingImageB::Name() { return "BranchingImageB"; }

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::BranchingImageC > { static const char* name() { return "BranchingImageC"; } };
template<> struct DataTypeName< defaulttype::BranchingImageUC > { static const char* name() { return "BranchingImageUC"; } };
template<> struct DataTypeName< defaulttype::BranchingImageI > { static const char* name() { return "BranchingImageI"; } };
template<> struct DataTypeName< defaulttype::BranchingImageUI > { static const char* name() { return "BranchingImageUI"; } };
template<> struct DataTypeName< defaulttype::BranchingImageS > { static const char* name() { return "BranchingImageS"; } };
template<> struct DataTypeName< defaulttype::BranchingImageUS > { static const char* name() { return "BranchingImageUS"; } };
template<> struct DataTypeName< defaulttype::BranchingImageL > { static const char* name() { return "BranchingImageL"; } };
template<> struct DataTypeName< defaulttype::BranchingImageUL > { static const char* name() { return "BranchingImageUL"; } };
template<> struct DataTypeName< defaulttype::BranchingImageF > { static const char* name() { return "BranchingImageF"; } };
template<> struct DataTypeName< defaulttype::BranchingImageD > { static const char* name() { return "BranchingImageD"; } };
template<> struct DataTypeName< defaulttype::BranchingImageB > { static const char* name() { return "BranchingImageB"; } };

/// \endcond



} // namespace defaulttype


} // namespace sofa


#endif // IMAGE_BranchingImage_H
