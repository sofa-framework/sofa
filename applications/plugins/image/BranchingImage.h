
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





/// A BranchingImage is an array (size of t) of maps.
/// Each map key corresponds to a pixel index (x,y,z) and key = z*sizex*sizey+y*sizex+x.
/// Each pixel corresponds to a SuperimposedVoxels, alias an array of ConnectionVoxel.
/// a ConnectionVoxel stores a value for each channels + its neighbours indices
template<typename _T>
struct BranchingImage
{

public:

    /// stored type
    typedef _T T;

    /// each direction around a voxel
    typedef enum { BACK=0, Zm1=BACK, BOTTOM=1, Ym1=BOTTOM, LEFT=2, Xm1=LEFT, FRONT=3, Zp1=FRONT, TOP=4, Yp1=TOP, RIGHT=5, Xp1=RIGHT, NB_NeighbourDirections=6 } NeighbourDirections;


    /// a ConnectionVoxel stores a value for each channels + its neighbour indices
    /// @todo are the indices a good thing or would a pointer be better?
    /// NB: a ConnectionVoxel does not know its spectrum size (nb of channels) to save memory
    class ConnectionVoxel
    {

    public:

        /// returns the opposite direction of a given direction  left->right,  right->left
        inline NeighbourDirections oppositeDirection( NeighbourDirections d ) { return (d+3)%NB_NeighbourDirections; }

        /// default constructor = no allocation
        ConnectionVoxel() : value(0) {}
        /// with allocation constructor
        ConnectionVoxel( size_t size ) { value = new T[size]; }
        ~ConnectionVoxel() { if( value ) delete [] value; }

        /// copy
        void clone( const ConnectionVoxel& cv, unsigned spectrum )
        {
            if( value ) delete [] value;
            value = new T[spectrum];
            memcpy( value, cv.value, spectrum*sizeof(T) );
            connections = cv.connections;
            index = cv.index;
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

        unsigned index; ///< the 1D position in the full image

        T* value; ///< value of the voxel for each channel (value is the size of the C dimension of the ConnectionImage)

        Vec< NB_NeighbourDirections, NoPreallocationVector<unsigned> > connections; ///< neighbours of the voxels. In each 6 directions (bottom, up, left...), a list of all connected voxels (indices in the Voxels list of the neighbour pixel in the ConnectionImage)

        /// accessor
        /// @warning index must be less than the spectrum
        T& operator [] (size_t index) const
        {
            return value[ index ];
        }

        bool isEqual( const ConnectionVoxel& other, unsigned spectrum ) const
        {
            for( unsigned i=0 ; i<spectrum ; ++i )
                if( value[i] != other.value[i] ) return false;
            return true;
        }

    private: // cannot be private to be able to compile std::container copy constructors...

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

        // copy constructor
        SuperimposedVoxels( const SuperimposedVoxels& cv, unsigned spectrum ) : Inherited()
        {
            this->resize( cv.size() );
            for( unsigned i=0 ; i<cv.size() ; ++i )
            {
                (*this)[i].clone( cv[i], spectrum );
            }
        }

        void push_back( const ConnectionVoxel& v, unsigned spectrum )
        {
            Inherited::resizeAndKeep( this->_size+1 );
            this->last().clone( v, spectrum );
        }

        void clone( const SuperimposedVoxels& other, unsigned spectrum )
        {
            resize( other._size );
            for( unsigned i=0 ; i<this->_size ; ++i )
            {
                this->_array[i].clone( other._array[i], spectrum );
            }
        }

        bool isEqual( const SuperimposedVoxels& other, unsigned spectrum ) const
        {
            if( this->_size != other._size ) return false;
            for( unsigned i=0 ; i<this->_size ; ++i )
                if( !this->_array[i].isEqual( other._array[i], spectrum ) ) return false;
            return true;
        }

        void toFlatVoxel( T& v, unsigned conversionType, unsigned channel ) const
        {
            if( this->empty() ) return;

            switch( conversionType )
            {
            case 0:
            default:
                v = this->_array[0][channel];
                break;
            }
        }


        /// @todo all needed operators +, +=, etc. can be overloaded here


    private :

        /// impossible to copy a ConnectedVoxel without the spectrum size
        void push_back( const ConnectionVoxel& v ) { assert(false); }
        /// copy constructor - impossible to copy a ConnectedVoxel without the spectrum size
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

        // possible overloads or helper functions

        void clone( const BranchingImage3D& other, unsigned spectrum )
        {
            resize( other._size );
            for( unsigned i=0 ; i<this->_size ; ++i )
            {
                this->_array[i].clone( other._array[i], spectrum );
            }
        }

        bool isEqual( const BranchingImage3D& other, unsigned spectrum ) const
        {
            if( this->_size != other._size ) return false;
            for( unsigned i=0 ; i<this->_size ; ++i )
                if( !this->_array[i].isEqual( other._array[i], spectrum ) ) return false;
            return true;
        }

    private:

        BranchingImage3D( const BranchingImage3D& ) { assert(false); }
        void operator=( const BranchingImage3D& ) { assert(false); }
        bool operator==( const BranchingImage3D& ) const { assert(false); }
        void push_back( const SuperimposedVoxels& v ) { assert(false); }

    }; // class BranchingImage




    typedef enum{ DIMENSION_X=0, DIMENSION_Y, DIMENSION_Z, DIMENSION_S /* spectrum = nb channels*/, DIMENSION_T /*4th dimension = time*/, NB_DimensionLabel } DimensionLabel;
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
                    ConnectionVoxel v( dimension[DIMENSION_S] );
//                    Voxel& v = imt[index1D].add( dimension[DIMENSION_S] );
                    for( unsigned c = 0 ; c<dimension[DIMENSION_S] ; ++c )
                        v[c] = cimgt(x,y,z,c);
                    v.index = index1D;
                    imt[index1D].push_back( v, dimension[DIMENSION_S] );
                    //imt.add( index1D, v, dimension[DIMENSION_S] );
                    // neighbours
                    if( x>0 && cimgt.get_vector_at(x-1,y,z).magnitude(1) != 0 )
                        v.connections[LEFT].push_back( 0 );
                    if( (unsigned)x<dimension[DIMENSION_X]-1 && cimgt.get_vector_at(x+1,y,z).magnitude(1) != 0 )
                        v.connections[RIGHT].push_back( 0 );
                    if( y>0 && cimgt.get_vector_at(x,y-1,z).magnitude(1) != 0 )
                        v.connections[BOTTOM].push_back( 0 );
                    if( (unsigned)y<dimension[DIMENSION_Y]-1 && cimgt.get_vector_at(x,y+1,z).magnitude(1) != 0 )
                        v.connections[TOP].push_back( 0 );
                    if( z>0 && cimgt.get_vector_at(x,y,z-1).magnitude(1) != 0 )
                        v.connections[BACK].push_back( 0 );
                    if( (unsigned)z<dimension[DIMENSION_Z]-1 && cimgt.get_vector_at(x,y,z+1).magnitude(1) != 0 )
                        v.connections[FRONT].push_back( 0 );
                }
                ++index1D;
            }
        }

        return *this;
    }


    /// conversion to a flat image
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

    /// compute the map key of a neighbour (supposing it is valid neighbour)
    inline unsigned index1DNeighbour( unsigned key, NeighbourDirections dir ) const
    {
        switch( dir )
        {
            case LEFT:   return key - 1;
            case RIGHT:  return key + 1;
            case BOTTOM: return key - dimension[DIMENSION_X];
            case TOP:    return key + dimension[DIMENSION_X];
            case BACK:   return key - sliceSize;
            case FRONT:  return key + sliceSize;
            default: return -1;
        }
    }


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


    inline friend std::istream& operator >> ( std::istream& in, BranchingImage<T>& im )
    {
        Dimension dim;
        in >> dim;
        im.setDimension( dim );
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const BranchingImage<T>& im )
    {
        out << im.getDimension();
        return out;
    }


    bool operator==( const BranchingImage<T>& other ) const
    {
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
            if( !imgList[t].isEqual( other.imgList[t], dimension[DIMENSION_S] ) ) return false;
        return true;
    }
    bool operator!=( const BranchingImage<T>& other ) const
    {
        return !(*this==other);
    }


    size_t approximativeSizeInBytes() const
    {
        size_t total = dimension[DIMENSION_T]*(imageSize+1)*( sizeof(unsigned) + sizeof(void*) ); // superimposed voxel vectors + BranchingImage3D vector

        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            const BranchingImage3D& imt = imgList[t];
            for( unsigned int index=0 ; index<imt.size() ; ++index )
            {
                total += imt[index].size()*( sizeof(unsigned) /*index*/ + dimension[DIMENSION_S]*sizeof(T) ); // voxel = index + all channels
                for( unsigned v=0 ; v<imt[index].size() ; ++v )
                {
                    total += NB_NeighbourDirections * ( sizeof(unsigned) + sizeof(void*) ); // 6 neighbour vectors per voxel
                    for( unsigned d=0 ; d<NB_NeighbourDirections ; ++d )
                    {
                        total += imt[index][v].connections[d].size() * sizeof( T* ); // neighbours
                    }
                }
            }
        }
        return total;
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
