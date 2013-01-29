
#ifndef IMAGE_SPARSEIMAGE_H
#define IMAGE_SPARSEIMAGE_H



#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <map>

#include "ImageTypes.h"

namespace sofa
{

namespace defaulttype
{

using helper::vector;


/// minimalist vector with a tight memory array
/// the particularity is not to pre-allocate (no unused memory is allocated without a user choice)
/// not efficient to add an entry (since it must resize the array)
/// @todo: move it somewhere in sofa::helper
template<class T>
class NoPreallocationVector
{

public:

    /// default constructor, no allocation, no initialization
    NoPreallocationVector() : _array(0), _size(0) {}
    /// default destructor
    ~NoPreallocationVector() { if( _array ) delete [] _array; }

    /// copy constructor
    NoPreallocationVector( const NoPreallocationVector<T>& c )
    {
        _array = new T[c._size];
        _size = c._size;
        memcpy( _array, c._array, _size*sizeof(T) );
    }

    /// clone
    void operator=( const NoPreallocationVector<T>& c )
    {
        if( _array ) delete [] _array;
        _array = new T[c._size];
        _size = c._size;
        memcpy( _array, c._array, _size*sizeof(T) );
    }

    /// add a entry at the end
    /// @warning this has bad performances, since resizing/reallocating the array is necessary
    void push_back( const T& v )
    {
        if( !_array ) // alloc
        {
            _array = new T[++_size];
        }
        else // realloc
        {
            T* tmp = new T[_size+1];
            memcpy( tmp, _array, _size*sizeof(T) );
            delete [] _array;
            _array = tmp;
            _array[_size] = v;
            ++_size;
        }
    }

    /// @warning data is lost
    void resize( size_t newSize )
    {
        _size = newSize;
        if( _array ) delete [] _array;
        _array = new T[_size];
    }

    /// still-existing data is preserved
    void resizeAndKeep( size_t newSize )
    {
        T* tmpArray = new T[newSize];
        memcpy( tmpArray, _array, std::min(newSize,_size)*sizeof(T) );
        _size = newSize;
        if( _array ) delete [] _array;
        _array = tmpArray;
    }

    /// entry accessor
    T& operator[]( size_t index ) { assert( index < _size ); return _array[ index ]; }
    /// entry const accessor
    const T& operator[]( size_t index ) const { assert( index < _size ); return _array[ index ]; }

    /// first entry accessor
    T& first() { assert(_size>0); return _array[0]; }
    /// first entry const accessor
    const T& first() const { assert(_size>0); return _array[0]; }
    /// last entry accessor
    T& last() { assert(_size>0); return _array[_size-1]; }
    /// last entry const accessor
    const T& last() const { assert(_size>0); return _array[_size-1]; }

    /// @returns the entry number (effective and allocated)
    inline size_t size() const { return _size; }
    /// @returns true iff the vector is empty (no entry)
    inline bool empty() const { return !_size; }

protected:

    T* _array; ///< the array where to store the entries
    size_t _size; ///< the array size

};




/// a ConnectionVoxel stores a value for each channels + its neighbour indices
/// @todo are the indices a good thing or would a pointer be better?
/// NB: a ConnectionVoxel does not know its spectrum size (nb of channels) to save memory
template<typename T>
struct ConnectionVoxel
{
    /// each direction around a voxel
    typedef enum { BACK=0, Zm1=BACK, BOTTOM=1, Ym1=BOTTOM, LEFT=2, Xm1=LEFT, FRONT=3, Zp1=FRONT, TOP=4, Yp1=TOP, RIGHT=5, Xp1=RIGHT, NB_NeighbourDirections=6 } NeighbourDirections;

    /// returns the opposite direction of a given direction  left->right,  right->left
    inline NeighbourDirections oppositeDirection( NeighbourDirections d ) { return (d+3)%NB_NeighbourDirections; }

    /// default constructor = no allocation
    ConnectionVoxel() : value(0) {}
    ~ConnectionVoxel() { if( value ) delete [] value; }

    /// copy
    void clone( const ConnectionVoxel& cv, unsigned spectrum )
    {
        if( value ) delete [] value;
        value = new T[spectrum];
        memcpy( value, cv.value, spectrum*sizeof(T) );
        connections = cv.connections;
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

    T* value; ///< value of the voxel for each channel (value is the size of the C dimension of the ConnectionImage)

    Vec< NB_NeighbourDirections, NoPreallocationVector<unsigned> > connections; ///< neighbours of the voxels. In each 6 directions (bottom, up, left...), a list of all connected voxels (indices in the Voxels list of the neighbour pixel in the ConnectionImage)

    /// accessor
    /// @warning index must be less than the spectrum
    T& operator [] (size_t index) const
    {
        return value[ index ];
    }

};



/// An array of ConnectionVoxel
template<typename T>
struct SuperimposedVoxels : public NoPreallocationVector< ConnectionVoxel<T> >
{
    typedef NoPreallocationVector< ConnectionVoxel<T> > Inherited;
    typedef ConnectionVoxel<T> Voxel;

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

    Voxel& add( unsigned spectrum /*nb channels*/ )
    {
        this->push_back( Voxel() );
        this->last().resize( spectrum );
        return this->last();
    }


    // all needed operators +, +=, etc. can be overloaded here


};



// would be more efficient with a hash_map/unordered_map but what about memory?
// is sparse necessary ? a dense 3D images of pointers does not seem so bad.
template<typename Voxels>
class BranchingImageT : public std::map<unsigned,Voxels>
{
    // possible overloads or helper functions
};




/// A SparseImage is an array (size of t) of maps.
/// Each map key corresponds to a pixel index (x,y,z) and key = z*sizex*sizey+y*sizex+x.
/// Each pixel corresponds to a SuperimposedVoxels, alias an array of ConnectionVoxel.
/// a ConnectionVoxel stores a value for each channels + its neighbours indices
template<typename _T>
struct SparseImage
{

public:

    typedef _T T;
    typedef SuperimposedVoxels<T> Voxels;
    typedef typename Voxels::Voxel Voxel;
    typedef typename Voxel::NeighbourDirections NeighbourDirections;

    typedef BranchingImageT<Voxels> BranchingImage;
    typedef typename BranchingImage::const_iterator BranchingImageCIt;
    typedef typename BranchingImage::iterator BranchingImageIt;


    typedef enum{ DIMENSION_X=0, DIMENSION_Y, DIMENSION_Z, DIMENSION_S /* spectrum = nb channels*/, DIMENSION_T /*4th dimension = time*/, NB_DimensionLabel } DimensionLabel;
    typedef Vec<NB_DimensionLabel,unsigned int> Dimension; // [x,y,z,s,t]



    Dimension dimension; ///< the image dimensions [x,y,z,s,t]
    unsigned sliceSize; ///< (x,y) slice size
    BranchingImage* imgList; ///< array of BranchingImage over time t



    static const char* Name();

    ///constructors/destructors
    SparseImage() : dimension(), imgList(0) {}
    ~SparseImage()
    {
        if( imgList ) delete [] imgList;
    }


    /// copy constructor
    SparseImage(const SparseImage<T>& img) : dimension(), imgList(0)
    {
        *this = img;
    }

    /// clone
    SparseImage<T>& operator=(const SparseImage<T>& im)
    {
        // allocate & copy everything
        setDimension( im.getDimension() );

        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            const BranchingImage& imt = im.imgList[t];
            BranchingImage& thisimt = imgList[t];

            // for each pixel of the given image
            for( BranchingImageCIt it = imt.begin() ; it != imt.end() ; ++it )
            {
                // at the same place (key=it->first), create a new Voxels with copy construction copying it->second
                thisimt[it->first] = Voxels( it->second );
            }
        }

        return *this;
    }


    /// conversion from flat image to connection image
    SparseImage(const Image<T>& img)
    {
        *this = img;
    }

    /// conversion from flat image to connection image
    SparseImage<T>& operator=(const Image<T>& im)
    {
        setDimension( im.getDimensions() );

        for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
        {
            BranchingImage& imt = imgList[t];
            const CImg<T>& cimgt = im.getCImg(t);
            unsigned index1D = 0;
            cimg_forXYZ(cimgt,x,y,z)
            {
                CImg<long double> vect=cimgt.get_vector_at(x,y,z);
                if( vect.magnitude(1) != 0 )
                {
                    assert( index1D == index3Dto1D(x,y,z) );
                    Voxel& v = imt[index1D].add( dimension[DIMENSION_S] );
                    for( unsigned c = 0 ; c<dimension[DIMENSION_S] ; ++c )
                        v[c] = cimgt(x,y,z,c);
                    // neighbours
                    if( x>0 && cimgt.get_vector_at(x-1,y,z).magnitude(1) != 0 )
                        v.connections[Voxel::LEFT].push_back( 0 );
                    if( (unsigned)x<dimension[DIMENSION_X]-1 && cimgt.get_vector_at(x+1,y,z).magnitude(1) != 0 )
                        v.connections[Voxel::RIGHT].push_back( 0 );
                    if( y>0 && cimgt.get_vector_at(x,y-1,z).magnitude(1) != 0 )
                        v.connections[Voxel::BOTTOM].push_back( 0 );
                    if( (unsigned)y<dimension[DIMENSION_Y]-1 && cimgt.get_vector_at(x,y+1,z).magnitude(1) != 0 )
                        v.connections[Voxel::TOP].push_back( 0 );
                    if( z>0 && cimgt.get_vector_at(x,y,z-1).magnitude(1) != 0 )
                        v.connections[Voxel::BACK].push_back( 0 );
                    if( (unsigned)z<dimension[DIMENSION_Z]-1 && cimgt.get_vector_at(x,y,z+1).magnitude(1) != 0 )
                        v.connections[Voxel::FRONT].push_back( 0 );
                }
                ++index1D;
            }
        }

        return *this;
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
            case Voxel::LEFT:   return key - 1;
            case Voxel::RIGHT:  return key + 1;
            case Voxel::BOTTOM: return key - dimension[DIMENSION_X];
            case Voxel::TOP:    return key + dimension[DIMENSION_X];
            case Voxel::BACK:   return key - sliceSize;
            case Voxel::FRONT:  return key + sliceSize;
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
        assert( newDimension[DIMENSION_T] > 0 );

        if( imgList ) delete [] imgList;
        dimension = newDimension;
        imgList = new BranchingImage[dimension[DIMENSION_T]];
        sliceSize = dimension[DIMENSION_X]*dimension[DIMENSION_Y];
    }




    /**
    * Returns histograms of image channels (mergeChannels=false) or a single histogram of the norm (mergeChannels=true)
    * the returned image size is [dimx,1,1,mergeChannels?1:nbChannels]
    * Returns min / max values
    * @warning: never tested :)
    */
    CImg<unsigned int> get_histogram(T& value_min, T& value_max, const unsigned int dimx, const bool mergeChannels=false) const
    {
        if( !imgList ) return CImg<unsigned int>();

        const unsigned int s=mergeChannels?1:dimension[DIMENSION_S];
        CImg<unsigned int> res(dimx,1,1,s,0);

        // considering there are always 0s
        value_min=0;
        value_max=0;

        if(mergeChannels)
        {
            for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
            {
                for( BranchingImageCIt it = imgList[t].begin() ; it != imgList[t].end() ; ++it )
                {
                    const Voxels& voxels = it->second;
                    for( unsigned v=0 ; v<voxels.size() ; ++v )
                    {
                        T tval = (T)voxels[v].magnitude(dimension[DIMENSION_S]);
                        if(value_min>tval) value_min=tval;
                        if(value_max<tval) value_max=tval;
                    }
                }
            }

            if(value_max==value_min) value_max=value_min+(T)1;

            for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
            {
                for( BranchingImageCIt it = imgList[t].begin() ; it != imgList[t].end() ; ++it )
                {
                    const Voxels& voxels = it->second;
                    for( unsigned v=0 ; v<voxels.size() ; ++v )
                    {
                        long double val = voxels[v].magnitude(dimension[DIMENSION_S]);
                        long double v = ((long double)val-(long double)value_min)/((long double)value_max-(long double)value_min)*((long double)(dimx-1));
                        if(v<0) v=0;
                        else if(!finite(v)) v=0;
                        else if(v>(long double)(dimx-1)) v=(long double)(dimx-1);
                        ++res((int)(v),0,0,0);
                    }
                }
            }
        }
        else
        {


            for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
            {
                for( BranchingImageCIt it = imgList[t].begin() ; it != imgList[t].end() ; ++it )
                {
                    const Voxels& voxels = it->second;
                    for( unsigned v=0 ; v<voxels.size() ; ++v )
                    {
                        T tvalmin = voxels[v].min(dimension[DIMENSION_S]);
                        T tvalmax = voxels[v].max(dimension[DIMENSION_S]);
                        if(value_min>tvalmin) value_min=tvalmin;
                        if(value_max<tvalmax) value_max=tvalmax;
                    }
                }
            }

            if(value_max==value_min) value_max=value_min+(T)1;

            for( unsigned t=0 ; t<dimension[DIMENSION_T] ; ++t )
            {
                for( BranchingImageCIt it = imgList[t].begin() ; it != imgList[t].end() ; ++it )
                {
                    const Voxels& voxels = it->second;
                    for( unsigned v=0 ; v<voxels.size() ; ++v )
                    {
                        for( unsigned c=0 ; c<dimension[DIMENSION_S] ; ++c )
                        {
                            const T val = voxels[v][c];
                            long double v = ((long double)val-(long double)value_min)/((long double)value_max-(long double)value_min)*((long double)(dimx-1));
                            if(v<0) v=0;
                            else if(!finite(v)) v=0;
                            else if(v>(long double)(dimx-1)) v=(long double)(dimx-1);
                            ++res((int)(v),0,0,c);
                        }
                    }
                }
            }
        }

        return res;
    }

//    // returns an image corresponing to a plane indexed by "coord" along "axis" and inside a bounding box
//    CImg<T> get_plane(const unsigned int coord,const unsigned int axis,const Mat<2,3,unsigned int>& ROI,const unsigned int t=0, const bool mergeChannels=false) const
//    {
//        if(mergeChannels)    return get_plane(coord,axis,ROI,t,false).norm();
//        else
//        {
//            if(axis==0)       return getCImg(t).get_crop(coord,ROI[0][1],ROI[0][2],0,coord,ROI[1][1],ROI[1][2],getCImg(t).spectrum()-1).permute_axes("zyxc");
//            else if(axis==1)  return getCImg(t).get_crop(ROI[0][0],coord,ROI[0][2],0,ROI[1][0],coord,ROI[1][2],getCImg(t).spectrum()-1).permute_axes("xzyc");
//            else              return getCImg(t).get_crop(ROI[0][0],ROI[0][1],coord,0,ROI[1][0],ROI[1][1],coord,getCImg(t).spectrum()-1);
//        }
//    }

//    // returns a binary image cutting through 3D input meshes, corresponding to a plane indexed by "coord" along "axis" and inside a bounding box
//    // positions are in image coordinates
//    template<typename Real>
//    CImg<bool> get_slicedModels(const unsigned int coord,const unsigned int axis,const Mat<2,3,unsigned int>& ROI,const ResizableExtVector<Vec<3,Real> >& position, const ResizableExtVector< component::visualmodel::VisualModelImpl::Triangle >& triangle, const ResizableExtVector< component::visualmodel::VisualModelImpl::Quad >& quad) const
//    {
//        const unsigned int dim[3]= {ROI[1][0]-ROI[0][0]+1,ROI[1][1]-ROI[0][1]+1,ROI[1][2]-ROI[0][2]+1};
//        CImg<bool> ret;
//        if(axis==0)  ret=CImg<bool>(dim[2],dim[1]);
//        else if(axis==1)  ret=CImg<bool>(dim[0],dim[2]);
//        else ret=CImg<bool>(dim[0],dim[1]);
//        ret.fill(false);

//        if(triangle.size()==0 && quad.size()==0) //pt visu
//        {
//            for (unsigned int i = 0; i < position.size() ; i++)
//            {
//                Vec<3,unsigned int> pt((unsigned int)round(position[i][0]),(unsigned int)round(position[i][1]),(unsigned int)round(position[i][2]));
//                if(pt[axis]==coord) if(pt[0]>=ROI[0][0] && pt[0]<=ROI[1][0]) if(pt[1]>=ROI[0][1] && pt[1]<=ROI[1][1])	if(pt[2]>=ROI[0][2] && pt[2]<=ROI[1][2])
//                            {
//                                if(axis==0)			ret(pt[2]-ROI[0][2],pt[1]-ROI[0][1])=true;
//                                else if(axis==1)	ret(pt[0]-ROI[0][0],pt[2]-ROI[0][2])=true;
//                                else				ret(pt[0]-ROI[0][0],pt[1]-ROI[0][1])=true;
//                            }

//            }
//        }
//        else
//        {
//            //unsigned int count;
//            Real alpha;
//            Vec<3,Real> v[4];
//            Vec<3,int> pt[4];
//            bool tru=true;

//            for (unsigned int i = 0; i < triangle.size() ; i++)  // box/ triangle intersection -> polygon with a maximum of 5 edges, to draw
//            {
//                for (unsigned int j = 0; j < 3 ; j++) { v[j] = position[triangle[i][j]]; pt[j]=Vec<3,int>((int)round(v[j][0]),(int)round(v[j][1]),(int)round(v[j][2])); }

//                vector<Vec<3,int> > pts;
//                for (unsigned int j = 0; j < 3 ; j++)
//                {
//                    if(pt[j][axis]==(int)coord) pts.push_back(pt[j]);
//                    unsigned int k=(j==2)?0:j+1;
//                    if(pt[j][axis]<pt[k][axis])
//                    {
//                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                    }
//                    else
//                    {
//                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                    }
//                }
//                for (unsigned int j = 0; j < pts.size() ; j++)
//                {
//                    unsigned int k=(j==pts.size()-1)?0:j+1;
//                    {
//                        if(axis==0)			ret.draw_line(pts[j][2]-(int)ROI[0][2],pts[j][1]-(int)ROI[0][1],pts[k][2]-(int)ROI[0][2],pts[k][1]-(int)ROI[0][1],&tru);
//                        else if(axis==1)	ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][2]-(int)ROI[0][2],pts[k][0]-(int)ROI[0][0],pts[k][2]-(int)ROI[0][2],&tru);
//                        else				ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][1]-(int)ROI[0][1],pts[k][0]-(int)ROI[0][0],pts[k][1]-(int)ROI[0][1],&tru);
//                    }
//                }

//            }
//            for (unsigned int i = 0; i < quad.size() ; i++)
//            {
//                for (unsigned int j = 0; j < 4 ; j++) { v[j] = position[quad[i][j]]; pt[j]=Vec<3,int>((int)round(v[j][0]),(int)round(v[j][1]),(int)round(v[j][2])); }

//                vector<Vec<3,int> > pts;
//                for (unsigned int j = 0; j < 4 ; j++)
//                {
//                    if(pt[j][axis]==(int)coord) pts.push_back(pt[j]);
//                    unsigned int k=(j==2)?0:j+1;
//                    if(pt[j][axis]<pt[k][axis])
//                    {
//                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                    }
//                    else
//                    {
//                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
//                    }
//                }
//                for (unsigned int j = 0; j < pts.size() ; j++)
//                {
//                    unsigned int k=(j==pts.size()-1)?0:j+1;
//                    {
//                        if(axis==0)			ret.draw_line(pts[j][2]-(int)ROI[0][2],pts[j][1]-(int)ROI[0][1],pts[k][2]-(int)ROI[0][2],pts[k][1]-(int)ROI[0][1],&tru);
//                        else if(axis==1)	ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][2]-(int)ROI[0][2],pts[k][0]-(int)ROI[0][0],pts[k][2]-(int)ROI[0][2],&tru);
//                        else				ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][1]-(int)ROI[0][1],pts[k][0]-(int)ROI[0][0],pts[k][1]-(int)ROI[0][1],&tru);
//                    }
//                }
//            }

//        }
//        return ret;
//    }

};


typedef SparseImage<char> SparseImageC;
typedef SparseImage<unsigned char> SparseImageUC;
typedef SparseImage<int> SparseImageI;
typedef SparseImage<unsigned int> SparseImageUI;
typedef SparseImage<short> SparseImageS;
typedef SparseImage<unsigned short> SparseImageUS;
typedef SparseImage<long> SparseImageL;
typedef SparseImage<unsigned long> SparseImageUL;
typedef SparseImage<float> SparseImageF;
typedef SparseImage<double> SparseImageD;
typedef SparseImage<bool> SparseImageB;

template<> inline const char* SparseImageC::Name() { return "SparseImageC"; }
template<> inline const char* SparseImageUC::Name() { return "SparseImageUC"; }
template<> inline const char* SparseImageI::Name() { return "SparseImageI"; }
template<> inline const char* SparseImageUI::Name() { return "SparseImageUI"; }
template<> inline const char* SparseImageS::Name() { return "SparseImageS"; }
template<> inline const char* SparseImageUS::Name() { return "SparseImageUS"; }
template<> inline const char* SparseImageL::Name() { return "SparseImageL"; }
template<> inline const char* SparseImageUL::Name() { return "SparseImageUL"; }
template<> inline const char* SparseImageF::Name() { return "SparseImageF"; }
template<> inline const char* SparseImageD::Name() { return "SparseImageD"; }
template<> inline const char* SparseImageB::Name() { return "SparseImageB"; }

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::SparseImageC > { static const char* name() { return "SparseImageC"; } };
template<> struct DataTypeName< defaulttype::SparseImageUC > { static const char* name() { return "SparseImageUC"; } };
template<> struct DataTypeName< defaulttype::SparseImageI > { static const char* name() { return "SparseImageI"; } };
template<> struct DataTypeName< defaulttype::SparseImageUI > { static const char* name() { return "SparseImageUI"; } };
template<> struct DataTypeName< defaulttype::SparseImageS > { static const char* name() { return "SparseImageS"; } };
template<> struct DataTypeName< defaulttype::SparseImageUS > { static const char* name() { return "SparseImageUS"; } };
template<> struct DataTypeName< defaulttype::SparseImageL > { static const char* name() { return "SparseImageL"; } };
template<> struct DataTypeName< defaulttype::SparseImageUL > { static const char* name() { return "SparseImageUL"; } };
template<> struct DataTypeName< defaulttype::SparseImageF > { static const char* name() { return "SparseImageF"; } };
template<> struct DataTypeName< defaulttype::SparseImageD > { static const char* name() { return "SparseImageD"; } };
template<> struct DataTypeName< defaulttype::SparseImageB > { static const char* name() { return "SparseImageB"; } };

/// \endcond



} // namespace defaulttype


} // namespace sofa


#endif // IMAGE_SPARSEIMAGE_H
