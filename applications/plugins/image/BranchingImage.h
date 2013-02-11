
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

/// type identifier, must be unique
static const int IMAGELABEL_BRANCHINGIMAGE = 1;

/// A BranchingImage is an array (size of t) of vectors (one vector per pixel (x,y,z)).
/// Each pixel corresponds to a SuperimposedVoxels, alias a vector of ConnectionVoxel.
/// a ConnectionVoxel stores a value for each channels + its neighbours indices
/// Nesme, Kry, Jeřábková, Faure, "Preserving Topology and Elasticity for Embedded Deformable Models", Siggraph09
template<typename _T>
class BranchingImage
{

public:

    /// stored type
    typedef _T T;

    /// type identifier, must be unique
    static const int label = IMAGELABEL_BRANCHINGIMAGE;

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

            if( !spectrum || !cv.value ) { value=0; return; }

            value = new T[spectrum];
            memcpy( value, cv.value, spectrum*sizeof(T) );
            //index = cv.index;
            neighbours = cv.neighbours;
        }

        /// copy only the topology and initialize value size acoording to spectrum
        template<typename T2>
        void cloneTopology( const typename BranchingImage<T2>::ConnectionVoxel& cv, unsigned spectrum, const T defaultValue=(T)0 )
        {
            if( value ) delete [] value;

            if( !spectrum || !cv.value ) { value=0; return; }

            value = new T[spectrum];
            for( unsigned i=0 ; i<spectrum ; ++i ) value[i]=defaultValue;

            //index = cv.index;
            neighbours = cv.neighbours;
        }


        /// alloc or realloc without keeping existing data and without initialization
        void resize( size_t newSize )
        {
            if( value ) delete [] value;
            if( !newSize ) { value=0; return; }
            value = new T[newSize];
            // could do a realloc keeping existing data
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
        typedef Vec< NB_NeighbourDirections, NeighboursInOneDirection > Neighbours; ///< @todo a bit of space could be saved by explicitly create 6 vectors leftNeighbours, rightNeighbours...
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
            if( other.empty() ) clear();
            else
            {
                Inherited::resize( other._size );
                for( unsigned i=0 ; i<this->_size ; ++i )
                {
                    this->_array[i].clone( other._array[i], spectrum );
                }
            }
        }

        /// copy only the topology and initialize value size acoording to spectrum
        template<typename T2>
        void cloneTopology( const typename BranchingImage<T2>::SuperimposedVoxels& other, unsigned spectrum , const T defaultValue=(T)0 )
        {
            if( other.empty() ) clear();
            else
            {
                Inherited::resize( other.size());
                for( unsigned i=0 ; i<this->_size ; ++i )
                {
                    this->_array[i].cloneTopology<T2>( other[i], spectrum, defaultValue );
                }
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
        /// conversionType : 0->first voxel, 1->average, 2->nb superimposed voxels, 3->sum
        template<class T2>
        void toFlatVoxel( T2& v, unsigned conversionType, unsigned channel ) const
        {
            if( this->empty() ) return;

            switch( conversionType )
            {
            case 1:
                assert( this->_array[0][channel] <= std::numeric_limits<T2>::max() );
                v = (T2)this->_array[0][channel];
                for( unsigned i=1 ; i<this->_size ; ++i )
                {
                    assert( v <= std::numeric_limits<T2>::max()-this->_array[i][channel] );
                    v += (T2)this->_array[i][channel];
                }
                v = (T2)( v / (float)this->_size );
                break;
            case 0:
                assert( this->_array[0][channel] <= std::numeric_limits<T2>::max() );
                v = (T2)this->_array[0][channel];
                break;
            case 3:
                assert( this->_array[0][channel] <= std::numeric_limits<T2>::max() );
                v = (T2)this->_array[0][channel];
                for( unsigned i=1 ; i<this->_size ; ++i )
                {
                    assert( v <= std::numeric_limits<T2>::max()-this->_array[i][channel] );
                    v += (T2)this->_array[i][channel];
                }
                break;
            case 2:
            default:
                assert( this->_size <= (unsigned)std::numeric_limits<T2>::max() );
                v = (T2)this->_size;
                break;
            }
        }

        void resize( size_t newSize, unsigned spectrum )
        {
            Inherited::resize( newSize );
            for( unsigned i=0 ; i<this->_size ; ++i )
                this->_array[i].resize( spectrum );
        }

        void fill( const ConnectionVoxel& v, unsigned spectrum )
        {
            for( unsigned i=0 ; i<this->_size ; ++i )
                this->_array[i].clone( v, spectrum );
        }


        /// all needed operators +, +=, etc. can be overloaded here


    private :

        /// impossible to copy a ConnectedVoxel without the spectrum size
        void push_back( const ConnectionVoxel& v ) { assert(false); }
        SuperimposedVoxels( const SuperimposedVoxels& cv ) { assert(false); }
        void operator=( const SuperimposedVoxels& ) { assert(false); }
        bool operator==( const SuperimposedVoxels& ) const { assert(false); }
        void resize( size_t ) { assert(false); }
        void fill( const T& ) { assert(false); }

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

        /// copy only the topology and initialize value size acoording to spectrum
        template<typename T2>
        void cloneTopology( const typename BranchingImage<T2>::BranchingImage3D& other, unsigned spectrum, const T defaultValue=(T)0)
        {
            resize( other.size() );
            for( unsigned i=0 ; i<this->_size ; ++i )
            {
                this->_array[i].cloneTopology<T2>( other[i], spectrum, defaultValue );
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
    typedef Dimension imCoord; // to have a common api with Image

    /// index indentifying a single connectionVoxel= index1D + SuperimposedIndex
    typedef std::pair<unsigned,unsigned> voxelIndex;

    Dimension dimension; ///< the image dimensions [x,y,z,s,t] - @todo maybe this could be implicit?
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
    template<class T2>
    BranchingImage<T>& operator=(const Image<T2>& im)
    {
        setDimension( im.getDimensions() );

        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            BranchingImage3D& imt = imgList[t];
            const CImg<T2>& cimgt = im.getCImg(t);
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
    /// conversionType : 0->first voxel, 1->average, 2->nb superimposed voxels, 3->sum
    template<class T2>
    void toImage( Image<T2>& img, unsigned conversionType ) const
    {
        img.clear();
        typename Image<T2>::imCoord dim = dimension;
        img.setDimensions( dim );
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            const BranchingImage3D& bimt = imgList[t];
            CImg<T2>& cimgt = img.getCImg(t);

            cimgt.fill((T2)0);

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
    inline unsigned getNeighbourIndex( NeighbourDirection d, unsigned index1D ) const
    {
        switch(d)
        {
        case LEFT: return index1D-1; break;
        case RIGHT: return index1D+1; break;
        case BOTTOM: return index1D-dimension[DIMENSION_X]; break;
        case TOP: return index1D+dimension[DIMENSION_X]; break;
        case BACK: return index1D-sliceSize; break;
        case FRONT: return index1D+sliceSize; break;
        default: return 0;
        }
    }


    /// \returns the list of connected neighbours of a given voxel
    void getNeighbours(std::vector< voxelIndex > &list, const voxelIndex& index, const unsigned t=0) const
    {
        list.clear();
        const ConnectionVoxel& voxel = this->imgList[t][index.first][index.second];
        const typename ConnectionVoxel::Neighbours& neighbours = voxel.neighbours;

        for( unsigned d = 0 ; d < NB_NeighbourDirections ; ++d )
        {
            const typename ConnectionVoxel::NeighboursInOneDirection& neighboursOneDirection = neighbours[d];
            for( unsigned n = 0 ; n < neighboursOneDirection.size() ; ++n )
            {
                unsigned neighbourIndex1d = getNeighbourIndex( (NeighbourDirection)d, index.first );
                list.push_back(voxelIndex(neighbourIndex1d,neighboursOneDirection[n]));
            }
        }
    }

    /// \returns the list of connected neighbours of a given voxel and the corresponding euclidean distances (image transform supposed to be linear)
    // TO DO: bias the distance using image values (= multiply distance by 2/(v1+v2))
    // TO DO: bias the distance using a lut (= multiply distance by lut(v1,v2))
    template<typename real>
    void getNeighboursAndDistances(std::vector< voxelIndex > &list, std::vector< real > &dist, const voxelIndex& index, const sofa::defaulttype::Vec<3,real>& voxelsize, const unsigned t=0) const
    {
        list.clear();
        dist.clear();

        const ConnectionVoxel& voxel = this->imgList[t][index.first][index.second];
        const typename ConnectionVoxel::Neighbours& neighbours = voxel.neighbours;

        for( unsigned d = 0 ; d < NB_NeighbourDirections ; ++d )
        {
            const typename ConnectionVoxel::NeighboursInOneDirection& neighboursOneDirection = neighbours[d];
            for( unsigned n = 0 ; n < neighboursOneDirection.size() ; ++n )
            {
                unsigned neighbourIndex1d = getNeighbourIndex( (NeighbourDirection)d, index.first );
                list.push_back(voxelIndex(neighbourIndex1d,neighboursOneDirection[n]));
                dist.push_back(voxelsize(d/2));
            }
        }
    }

    /// \returns image value at a given voxel index, time and channel
    /// @warning validity of indices, channel and time not checked
    inline const T& getValue(const unsigned& off1D, const unsigned& v, const unsigned c=0, const unsigned t=0) const  { return this->imgList[t][off1D][v].value[c]; }
    inline T& getValue(const unsigned& off1D, const unsigned& v, const unsigned c=0, const unsigned t=0) { return this->imgList[t][off1D][v].value[c]; }
    inline const T& getValue(const voxelIndex& index, const unsigned c=0, const unsigned t=0) const  { return this->imgList[t][index.first][index.second].value[c]; }
    inline T& getValue(const voxelIndex& index, const unsigned c=0, const unsigned t=0) { return this->imgList[t][index.first][index.second].value[c]; }

    /// \returns the direction between two neighbour voxels
    /// @warnings the two given voxels are supposed to be neighbours, otherwise NB_NeighbourDirections is returned
    /// example: returning LEFT means neighbourIndex is at the LEFT position of index
    inline NeighbourDirection getDirection( unsigned index1D, unsigned neighbourIndex1D ) const
    {
        int offset = neighbourIndex1D - index1D;
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
    const Dimension& getDimensions() const
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

    /// count the nb of sumperimposed voxels
    unsigned count() const
    {
        unsigned total = 0;
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            int index1d = 0;
            for( unsigned z=0 ; z<dimension[DIMENSION_Z] ; ++z )
                for( unsigned y=0 ; y<dimension[DIMENSION_Y] ; ++y )
                    for( unsigned x=0 ; x<dimension[DIMENSION_X] ; ++x )
                    {
                        total += imgList[t][index1d].size();
                        ++index1d;
                    }
        }
        return total;
    }


    /// sum every values (every channels)
    T sum() const
    {
        T total = 0;
        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            int index1d = 0;
            for( unsigned z=0 ; z<dimension[DIMENSION_Z] ; ++z )
                for( unsigned y=0 ; y<dimension[DIMENSION_Y] ; ++y )
                    for( unsigned x=0 ; x<dimension[DIMENSION_X] ; ++x )
                    {
                        for( unsigned v=0 ; v<imgList[t][index1d].size() ; ++v )
                            for( unsigned s=0 ; s<dimension[DIMENSION_S] ; ++s )
                                total += imgList[t][index1d][v][s];
                        ++index1d;
                    }
        }
        return total;
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
                            unsigned totalNeighbours = 0;
                            for( unsigned d = 0 ; d < NB_NeighbourDirections ; ++d )
                            {
                                const typename ConnectionVoxel::NeighboursInOneDirection& neighboursOneDirection = neighbours[d];
                                totalNeighbours += neighboursOneDirection.size();
                                for( unsigned n = 0 ; n < neighboursOneDirection.size() ; ++n )
                                {
                                    unsigned neighbourIndex = getNeighbourIndex( (NeighbourDirection)d, index1d );
                                    if( neighboursOneDirection[n] >= imt[neighbourIndex].size() )  return 1; // there is nobody where there should be the neighbour
                                    if( !imt[neighbourIndex][neighboursOneDirection[n]].isNeighbour( oppositeDirection((NeighbourDirection)d), imt.getOffset(index1d,voxels[v]) ) ) return 2; // complementary neighbour is no inserted
                                    //                                    if( imt[neighbourIndex][neighboursOneDirection[n]].index != neighbourIndex || voxels[v].index != index1d ) return false; // a voxel has a good index
                                }
                            }
                            if( !totalNeighbours ) return 3;
                        }
                        ++ index1d;
                    }
                }
            }
        }
        return 0;
    }
};


/// macros for image loops

#define bimg_for1(bound,i) for (unsigned i = 0; i<bound; ++i)
#define bimg_forC(img,c) bimg_for1(img.dimension[img.DIMENSION_S],c)
#define bimg_forX(img,x) bimg_for1(img.dimension[img.DIMENSION_X],x)
#define bimg_forY(img,y) bimg_for1(img.dimension[img.DIMENSION_Y],y)
#define bimg_forZ(img,z) bimg_for1(img.dimension[img.DIMENSION_Z],z)
#define bimg_forT(img,t) bimg_for1(img.dimension[img.DIMENSION_T],t)
#define bimg_foroff1D(img,off1D)  bimg_for1(img.imageSize,off1D)
#define bimg_forXY(img,x,y) bimg_forY(img,y) bimg_forX(img,x)
#define bimg_forXZ(img,x,z) bimg_forZ(img,z) bimg_forX(img,x)
#define bimg_forYZ(img,y,z) bimg_forZ(img,z) bimg_forY(img,y)
#define bimg_forXT(img,x,t) bimg_forT(img,t) bimg_forX(img,x)
#define bimg_forYT(img,y,t) bimg_forT(img,t) bimg_forY(img,y)
#define bimg_forZT(img,z,t) bimg_forT(img,t) bimg_forZ(img,z)
#define bimg_forXYZ(img,x,y,z) bimg_forZ(img,z) bimg_forXY(img,x,y)
#define bimg_forXYT(img,x,y,t) bimg_forT(img,t) bimg_forXY(img,x,y)
#define bimg_forXZT(img,x,z,t) bimg_forT(img,t) bimg_forXZ(img,x,z)
#define bimg_forYZT(img,y,z,t) bimg_fort(img,t) bimg_forYZ(img,y,z)
#define bimg_forXYZT(img,x,y,z,t) bimg_forT(img,t) bimg_forXYZ(img,x,y,z)
#define bimg_forVXYZT(img,v,x,y,z,t) bimg_forT(img,t) bimg_forXYZ(img,x,y,z) for( unsigned v=0 ; v<img.imgList[t][img.index3Dto1D(x,y,z)].size() ; ++v )
#define bimg_forCVXYZT(img,c,v,x,y,z,t) bimg_forT(img,t) bimg_forXYZ(img,x,y,z) for( unsigned v=0 ; v<img.imgList[t][img.index3Dto1D(x,y,z)].size() ; ++v ) bimg_forC(img,c)
#define bimg_forVoffT(img,v,off1D,t) bimg_forT(img,t) bimg_foroff1D(img,off1D) for( unsigned v=0 ; v<img.imgList[t][off1D].size() ; ++v )
#define bimg_forCVoffT(img,c,v,off1D,t) bimg_forT(img,t) bimg_foroff1D(img,off1D) for( unsigned v=0 ; v<img.imgList[t][off1D].size() ; ++v )  bimg_forC(img,c)


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
