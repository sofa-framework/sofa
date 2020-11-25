#pragma once 

#include <CImgPlugin/SOFACImg.h>

// datatypes
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>

// visual dependencies
#include <SofaBaseVisual/VisualModelImpl.h>
#include <SofaBaseVisual/VisualStyle.h>

// helpers
#include <sofa/helper/rmath.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/rmath.h>


namespace sofa
{

namespace defaulttype
{



/// a virtual, non templated Image class that can be allocated without knowing its exact type
struct BaseImage
{
    typedef Vec<5,unsigned int> imCoord; // [x,y,z,s,t]
    virtual void setDimensions(const imCoord& dim) = 0;
    virtual void fill(const SReal val)=0;
    virtual ~BaseImage() {}
};




//-----------------------------------------------------------------------------------------------//
/// 5d-image structure on top of a shared memory CImgList
//-----------------------------------------------------------------------------------------------//


template<typename _T>
struct Image : public BaseImage
{
    typedef _T T;
    typedef cimg_library::CImg<T> CImgT;

    /// the 5 dimension labels of an image ( x, y, z, spectrum=nb channels , time )
    typedef enum{ DIMENSION_X=0, DIMENSION_Y, DIMENSION_Z, DIMENSION_S /* spectrum = nb channels*/, DIMENSION_T /*4th dimension = time*/, NB_DimensionLabel } DimensionLabel;

protected:
    cimg_library::CImgList<T> img; // list of images along temporal dimension. Each image is 4-dimensional (x,y,z,s) where s is the spectrum (e.g. channels for color images, vector or tensor values, etc.)

public:
    static const char* Name();

    ///constructors/destructors
    Image() {}
    Image(const Image<T>& _img, bool shared=false) : img(_img.getCImgList(), shared) {}
    Image( const cimg_library::CImg<T>& _img ) : img(_img) {}

    /// copy operators
    Image<T>& operator=(const Image<T>& im)
    {
        if(im.getCImgList().size()) img.assign(im.getCImgList());
        return *this;
    }
    Image<T>& assign(const Image<T>& im, const bool shared=false)
    {
        if(im.getCImgList().size()) img.assign(im.getCImgList(),shared);
        return *this;
    }


    void clear() { img.assign(); }
    ~Image() override { clear(); }

    //accessors
    cimg_library::CImgList<T>& getCImgList() { return img; }
    const cimg_library::CImgList<T>& getCImgList() const { return img; }

    cimg_library::CImg<T>& getCImg(const unsigned int t=0) {
        if (t>=img.size())   {
            assert(img._data != nullptr);
            return *img._data;
        }
        return img(t);
    }
    const cimg_library::CImg<T>& getCImg(const unsigned int t=0) const {
        if (t>=img.size())   {
            assert(img._data != nullptr);
            return *img._data;
        }
        return img(t);
    }

    inline bool isEmpty() const {return img.size()==0;}

    /// check if image coordinates are inside bounds
    template<class t>
    inline bool isInside( t x, t y, t z ) const
    {
        if(isEmpty()) return false;
        if(x<0) return false;
        if(y<0) return false;
        if(z<0) return false;
        if(x>=(t)img(0).width()) return false;
        if(y>=(t)img(0).height()) return false;
        if(z>=(t)img(0).depth()) return false;
        return true;
    }

    imCoord getDimensions() const
    {
        imCoord dim;
        if(!img.size()) dim.fill(0);
        else
        {
            dim[0]=img(0).width();
            dim[1]=img(0).height();
            dim[2]=img(0).depth();
            dim[3]=img(0).spectrum();
            dim[4]=img.size();
        }
        return dim;
    }

    //affectors
    void setDimensions(const imCoord& dim) override
    {
        cimglist_for(img,l) img(l).resize(dim[0],dim[1],dim[2],dim[3]);
        if(img.size()>dim[4]) img.remove(dim[4],img.size()-1);
        else if(img.size()<dim[4]) img.insert(dim[4]-img.size(),cimg_library::CImg<T>(dim[0],dim[1],dim[2],dim[3]));
    }

    void fill(const SReal val) override
    {
        cimglist_for(img,l) img(l).fill((T)val);
    }

    //iostream
    inline friend std::istream& operator >> ( std::istream& in, Image<T>& im )
    {
        imCoord dim;  in>>dim;
        im.setDimensions(dim);
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const Image<T>& im )
    {
        out<<im.getDimensions();
        return out;
    }

    bool operator == ( const Image<T>& other ) const
    {
        if( img.size() != other.img.size() ) return false;
        for( unsigned t=0 ; t<img.size() ; ++t )
            if( img[t] != other.img[t] ) return false;
        return true;
    }

    bool operator != ( const Image<T>& other ) const
    {
        return !(*this==other);
    }



    //basic functions to complement CImgList


    /**
    * Returns histograms of image channels (mergeChannels=false) or a single histogram of the norm (mergeChannels=true)
    * the returned image size is [dimx,1,1,mergeChannels?1:nbChannels]
    * Returns min / max values
    */
    cimg_library::CImg<unsigned int> get_histogram(T& value_min, T& value_max, const unsigned int dimx, const bool mergeChannels=false) const
    {
        if(!img.size()) return cimg_library::CImg<unsigned int>();
        const unsigned int s=mergeChannels?1:img(0).spectrum();
        cimg_library::CImg<unsigned int> res(dimx,1,1,s,0);

        if(mergeChannels)
        {
            value_min=cimg_library::cimg::type<T>::max();
            value_max=cimg_library::cimg::type<T>::min();
            cimglist_for(img,l)
                    cimg_forXYZ(img(l),x,y,z)
            {
                cimg_library::CImg<long double> vect=img(l).get_vector_at(x,y,z);
                long double val=vect.magnitude();
                T tval=(T)val;
                if(value_min>tval) value_min=tval;
                if(value_max<tval) value_max=tval;
            }
            if(value_max==value_min) value_max=value_min+(T)1;
            cimglist_for(img,l)
                    cimg_forXYZ(img(l),x,y,z)
            {
                cimg_library::CImg<long double> vect=img(l).get_vector_at(x,y,z);
                long double val=vect.magnitude();
                long double v = ((long double)val-(long double)value_min)/((long double)value_max-(long double)value_min)*((long double)(dimx-1));
                if(v<0) v=0;
                else if(v>(long double)(dimx-1)) v=(long double)(dimx-1);
                ++res((int)(v),0,0,0);
            }
        }
        else
        {
            value_min=img.min();
            value_max=img.max();
            if(value_max==value_min) value_max=value_min+(T)1;
            cimglist_for(img,l)
                    cimg_forXYZC(img(l),x,y,z,c)
            {
                if((long double)value_max-(long double)value_min !=0)
                {
                    const T val = img(l)(x,y,z,c);
                    long double v = ((long double)val-(long double)value_min)/((long double)value_max-(long double)value_min)*((long double)(dimx-1));
                    if(v<0) v=0;
                    else if(v>(long double)(dimx-1)) v=(long double)(dimx-1);
                    ++res((int)(v),0,0,c);
                }
            }
        }
        return res;
    }

    // returns an image corresponing to a plane indexed by "coord" along "axis" and inside a bounding box
    cimg_library::CImg<T> get_plane(const unsigned int coord,const unsigned int axis,const Mat<2,3,unsigned int>& ROI,const unsigned int t=0, const bool mergeChannels=false) const
    {
        if(mergeChannels)    return get_plane(coord,axis,ROI,t,false).norm();
        else
        {
            if(axis==0)       return getCImg(t).get_crop(coord,ROI[0][1],ROI[0][2],0,coord,ROI[1][1],ROI[1][2],getCImg(t).spectrum()-1).permute_axes("zyxc");
            else if(axis==1)  return getCImg(t).get_crop(ROI[0][0],coord,ROI[0][2],0,ROI[1][0],coord,ROI[1][2],getCImg(t).spectrum()-1).permute_axes("xzyc");
            else              return getCImg(t).get_crop(ROI[0][0],ROI[0][1],coord,0,ROI[1][0],ROI[1][1],coord,getCImg(t).spectrum()-1);
        }
    }

    // returns a binary image cutting through 3D input meshes, corresponding to a plane indexed by "coord" along "axis" and inside a bounding box
    // positions are in image coordinates
    template<typename Real>
    cimg_library::CImg<bool> get_slicedModels(const unsigned int coord,const unsigned int axis,const Mat<2,3,unsigned int>& ROI,const helper::vector<Vec<3,Real> >& position, const helper::vector< component::visualmodel::VisualModelImpl::VisualTriangle >& triangle, const helper::vector< component::visualmodel::VisualModelImpl::VisualQuad >& quad) const
    {
        const unsigned int dim[3]= {ROI[1][0]-ROI[0][0]+1,ROI[1][1]-ROI[0][1]+1,ROI[1][2]-ROI[0][2]+1};
        cimg_library::CImg<bool> ret;
        if(axis==0)  ret=cimg_library::CImg<bool>(dim[2],dim[1]);
        else if(axis==1)  ret=cimg_library::CImg<bool>(dim[0],dim[2]);
        else ret=cimg_library::CImg<bool>(dim[0],dim[1]);
        ret.fill(false);

        if(triangle.size()==0 && quad.size()==0) //pt visu
        {
            for (unsigned int i = 0; i < position.size() ; i++)
            {
                Vec<3,unsigned int> pt((unsigned int)helper::round(position[i][0]),(unsigned int)helper::round(position[i][1]),(unsigned int)helper::round(position[i][2]));
                if(pt[axis]==coord) if(pt[0]>=ROI[0][0] && pt[0]<=ROI[1][0]) if(pt[1]>=ROI[0][1] && pt[1]<=ROI[1][1])	if(pt[2]>=ROI[0][2] && pt[2]<=ROI[1][2])
                {
                    if(axis==0)			ret(pt[2]-ROI[0][2],pt[1]-ROI[0][1])=true;
                    else if(axis==1)	ret(pt[0]-ROI[0][0],pt[2]-ROI[0][2])=true;
                    else				ret(pt[0]-ROI[0][0],pt[1]-ROI[0][1])=true;
                }

            }
        }
        else
        {
            //unsigned int count;
            Real alpha;
            Vec<3,Real> v[4];
            Vec<3,int> pt[4];
            bool tru=true;

            for (unsigned int i = 0; i < triangle.size() ; i++)  // box/ triangle intersection -> polygon with a maximum of 5 edges, to draw
            {
                for (unsigned int j = 0; j < 3 ; j++) { v[j] = position[triangle[i][j]]; pt[j]=Vec<3,int>((int)helper::round(v[j][0]),(int)helper::round(v[j][1]),(int)helper::round(v[j][2])); }

                helper::vector<Vec<3,int> > pts;
                for (unsigned int j = 0; j < 3 ; j++)
                {
                    if(pt[j][axis]==(int)coord) pts.push_back(pt[j]);
                    unsigned int k=(j==2)?0:j+1;
                    if(pt[j][axis]<pt[k][axis])
                    {
                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                    }
                    else
                    {
                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                    }
                }
                for (unsigned int j = 0; j < pts.size() ; j++)
                {
                    unsigned int k=(j==pts.size()-1)?0:j+1;
                    {
                        if(axis==0)			ret.draw_line(pts[j][2]-(int)ROI[0][2],pts[j][1]-(int)ROI[0][1],pts[k][2]-(int)ROI[0][2],pts[k][1]-(int)ROI[0][1],&tru);
                        else if(axis==1)	ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][2]-(int)ROI[0][2],pts[k][0]-(int)ROI[0][0],pts[k][2]-(int)ROI[0][2],&tru);
                        else				ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][1]-(int)ROI[0][1],pts[k][0]-(int)ROI[0][0],pts[k][1]-(int)ROI[0][1],&tru);
                    }
                }

            }
            for (unsigned int i = 0; i < quad.size() ; i++)
            {
                for (unsigned int j = 0; j < 4 ; j++) { v[j] = position[quad[i][j]]; pt[j]=Vec<3,int>((int)helper::round(v[j][0]),(int)helper::round(v[j][1]),(int)helper::round(v[j][2])); }

                helper::vector<Vec<3,int> > pts;
                for (unsigned int j = 0; j < 4 ; j++)
                {
                    if(pt[j][axis]==(int)coord) pts.push_back(pt[j]);
                    unsigned int k=(j==3)?0:j+1;
                    if(pt[j][axis]<pt[k][axis])
                    {
                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                    }
                    else
                    {
                        alpha=((Real)coord+0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                        alpha=((Real)coord-0.5 -v[k][axis])/(v[j][axis]-v[k][axis]); if( alpha>=0 &&  alpha <=1)  pts.push_back(Vec<3,int>((int)helper::round(v[j][0]*alpha + v[k][0]*(1.0-alpha)),(int)helper::round(v[j][1]*alpha + v[k][1]*(1.0-alpha)),(int)helper::round(v[j][2]*alpha + v[k][2]*(1.0-alpha))));
                    }
                }
                for (unsigned int j = 0; j < pts.size() ; j++)
                {
                    unsigned int k=(j==pts.size()-1)?0:j+1;
                    {
                        if(axis==0)			ret.draw_line(pts[j][2]-(int)ROI[0][2],pts[j][1]-(int)ROI[0][1],pts[k][2]-(int)ROI[0][2],pts[k][1]-(int)ROI[0][1],&tru);
                        else if(axis==1)	ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][2]-(int)ROI[0][2],pts[k][0]-(int)ROI[0][0],pts[k][2]-(int)ROI[0][2],&tru);
                        else				ret.draw_line(pts[j][0]-(int)ROI[0][0],pts[j][1]-(int)ROI[0][1],pts[k][0]-(int)ROI[0][0],pts[k][1]-(int)ROI[0][1],&tru);
                    }
                }
            }

        }
        return ret;
    }

    /// \returns an approximative size in bytes, useful for debugging
    size_t approximativeSizeInBytes() const
    {
        if( img.is_empty() ) return 0;
        return img(0).width()*img(0).height()*img(0).depth()*img(0).spectrum()*img.size()*sizeof(T);
    }

};


typedef Image<char> ImageC;
typedef Image<unsigned char> ImageUC;
typedef Image<int> ImageI;
typedef Image<unsigned int> ImageUI;
typedef Image<short> ImageS;
typedef Image<unsigned short> ImageUS;
typedef Image<long> ImageL;
typedef Image<unsigned long> ImageUL;
typedef Image<float> ImageF;
typedef Image<double> ImageD;
typedef Image<bool> ImageB;
typedef Image<SReal> ImageR;

template<> inline const char* ImageC::Name() { return "ImageC"; }
template<> inline const char* ImageUC::Name() { return "ImageUC"; }
template<> inline const char* ImageI::Name() { return "ImageI"; }
template<> inline const char* ImageUI::Name() { return "ImageUI"; }
template<> inline const char* ImageS::Name() { return "ImageS"; }
template<> inline const char* ImageUS::Name() { return "ImageUS"; }
template<> inline const char* ImageL::Name() { return "ImageL"; }
template<> inline const char* ImageUL::Name() { return "ImageUL"; }
template<> inline const char* ImageF::Name() { return "ImageF"; }
template<> inline const char* ImageD::Name() { return "ImageD"; }
template<> inline const char* ImageB::Name() { return "ImageB"; }

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::ImageC > { static const char* name() { return "ImageC"; } };
template<> struct DataTypeName< defaulttype::ImageUC > { static const char* name() { return "ImageUC"; } };
template<> struct DataTypeName< defaulttype::ImageI > { static const char* name() { return "ImageI"; } };
template<> struct DataTypeName< defaulttype::ImageUI > { static const char* name() { return "ImageUI"; } };
template<> struct DataTypeName< defaulttype::ImageS > { static const char* name() { return "ImageS"; } };
template<> struct DataTypeName< defaulttype::ImageUS > { static const char* name() { return "ImageUS"; } };
template<> struct DataTypeName< defaulttype::ImageL > { static const char* name() { return "ImageL"; } };
template<> struct DataTypeName< defaulttype::ImageUL > { static const char* name() { return "ImageUL"; } };
template<> struct DataTypeName< defaulttype::ImageF > { static const char* name() { return "ImageF"; } };
template<> struct DataTypeName< defaulttype::ImageD > { static const char* name() { return "ImageD"; } };
template<> struct DataTypeName< defaulttype::ImageB > { static const char* name() { return "ImageB"; } };

/// \endcond

//-----------------------------------------------------------------------------------------------//
// Transforms between image and space/time
//-----------------------------------------------------------------------------------------------//

// generic interface
template<typename _Real>
struct ImageTransform
{
    typedef _Real Real;
    typedef Vec<3,Real> Coord;		// 3d coords

public:
    virtual Coord fromImage(const Coord& ip) const {return ip;} // image coord to space transform
    virtual Real fromImage(const Real& ip) const {return ip;}	// image index to time transform
    virtual Coord toImage(const Coord& p) const {return p;}		// space coord to image transform
    virtual Real toImage(const Real& p) const {return p;}		// time to image index transform
    virtual Coord toImageInt(const Coord& p) const { Coord p2 = toImage(p); return Coord( helper::round(p2.x()),helper::round(p2.y()),helper::round(p2.z()) );}		// space coord to rounded image transform
    virtual Real toImageInt(const Real& p) const { return helper::round(toImage(p));}		// time to rounded image index transform

    virtual const Coord& getTranslation() const = 0;
    virtual const Coord& getRotation() const = 0;
    virtual const Coord& getScale() const = 0;

    virtual void update()=0;

};


// abstract class with vector and iostream
template<int N,typename _Real>
struct TImageTransform : public ImageTransform<_Real>
{
    typedef ImageTransform<_Real> Inherited;
    typedef typename Inherited::Real Real;
    typedef typename Inherited::Coord Coord;
    enum {size = N};
    typedef Vec<size,Real> Params;	// transform parameters

protected:
    Params P;

public:
    TImageTransform():P() { P.clear(); }
    TImageTransform(const Params& _P):P(_P) {}
    TImageTransform(const TImageTransform& T):P(T.getParams()) {}

    Params& getParams() {return P;}
    const Params& getParams() const {return P;}

    static const char* Name();

    inline friend std::istream& operator >> ( std::istream& in, TImageTransform<N,Real>& T ) { in>>T.getParams();	return in; }
    friend std::ostream& operator << ( std::ostream& out, const TImageTransform<N,Real>& T )	{ out<<T.getParams();	return out;	}

    TImageTransform& operator=(const TImageTransform& T) 	{ P=T.getParams(); this->update(); return *this; }
    void set(const Params& _P) { P=_P; 	this->update(); }
};



//  implementation of linear (scale+rotation+translation) and perspective transforms
//  for perspective transforms (only for 2D images), the pinhole camera is located at ( scalex(dimx-1)/2, scaley(dimy-1)/2, -scalez/2)

template<typename _Real>
struct ImageLPTransform : public TImageTransform<12,_Real>
{
    typedef TImageTransform<12,_Real> Inherited; // 12 params : translations,rotations,scales,timeOffset,timeScale,isPerspective
    typedef typename Inherited::Real Real;
    typedef typename Inherited::Params Params;
    typedef typename Inherited::Coord Coord;

protected:
    Real camx;	Real camy; // used only for perpective transforms (camera offset = c_x and c_y pinhole camera intrinsic parameters)

public:
    Coord& getTranslation() { return *reinterpret_cast<Coord*>(&this->P[0]); }
    const Coord& getTranslation() const { return *reinterpret_cast<const Coord*>(&this->P[0]); }
    Coord& getRotation() { return *reinterpret_cast<Coord*>(&this->P[3]); }
    const Coord& getRotation() const { return *reinterpret_cast<const Coord*>(&this->P[3]); }
    Coord& getScale() { return *reinterpret_cast<Coord*>(&this->P[6]); }
    const Coord& getScale() const { return *reinterpret_cast<const Coord*>(&this->P[6]); }
    Real& getOffsetT() { return *reinterpret_cast<Real*>(&this->P[9]); }
    const Real& getOffsetT() const { return *reinterpret_cast<const Real*>(&this->P[9]); }
    Real& getScaleT() { return *reinterpret_cast<Real*>(&this->P[10]); }
    const Real& getScaleT() const { return *reinterpret_cast<const Real*>(&this->P[10]);  }
    Real& isPerspective() { return *reinterpret_cast<Real*>(&this->P[11]); }
    const Real& isPerspective() const { return *reinterpret_cast<const Real*>(&this->P[11]); }

    ImageLPTransform()	// identity
        :Inherited()
    {
        getScale()[0]=getScale()[1]=getScale()[2]=getScaleT()=(Real)1.0;
        camx = camy = (Real)0.0;
    }

    virtual ~ImageLPTransform() {}

    //internal data
    helper::Quater<Real> qrotation; Coord axisrotation; Real phirotation; // "rotation" in other formats

    void setCamPos(const Real& cx,const Real& cy) {this->camx=cx;  this->camy=cy; }

    //internal data update
    virtual void update()
    {
        Coord rot=getRotation() * (Real)M_PI / (Real)180.0;
        qrotation = helper::Quater< Real >::createQuaterFromEuler(rot);
        qrotation.quatToAxis(axisrotation,phirotation);
        phirotation*=(Real)180.0/ (Real)M_PI;
    }

    //transform functions
    // note: for perpective transforms (f_x and f_y pinhole camera intrinsic parameters are scalez/2*scalex and scalez/2*scaley)
    virtual Coord fromImage(const Coord& ip) const
    {
        if(isPerspective()==0) return qrotation.rotate( ip.linearProduct(getScale()) ) + getTranslation();
        else if(isPerspective()==1)
        {
            Coord sp=ip.linearProduct(getScale());
            sp[0]+=(Real)2.0*ip[2]*getScale()[0]*(ip[0]-camx);
            sp[1]+=(Real)2.0*ip[2]*getScale()[1]*(ip[1]-camy);
            return qrotation.rotate( sp ) + getTranslation();
        }
        else if(isPerspective()==2) // half perspective, half orthographic
        {
            Coord sp=ip.linearProduct(getScale());
            sp[0]+=(Real)2.0*ip[2]*getScale()[0]*(ip[0]-camx);
            return qrotation.rotate( sp ) + getTranslation();
        }
        else // half perspective, half orthographic
        {
            Coord sp=ip.linearProduct(getScale());
            sp[1]+=(Real)2.0*ip[2]*getScale()[1]*(ip[1]-camy);
            return qrotation.rotate( sp ) + getTranslation();
        }
    }
    virtual Real fromImage(const Real& ip) const	{ return ip*getScaleT() + getOffsetT(); }
    virtual Coord toImage(const Coord& p) const
    {
        if(isPerspective()==0) return qrotation.inverseRotate( p-getTranslation() ).linearDivision(getScale());
        else if(isPerspective()==1)
        {
            Coord sp=qrotation.inverseRotate( p-getTranslation() );
            sp[0]=(sp[0]/getScale()[0] + (Real)2.0*sp[2]*camx/getScale()[2])/((Real)1.0 + (Real)2.0*sp[2]/getScale()[2]);
            sp[1]=(sp[1]/getScale()[1] + (Real)2.0*sp[2]*camy/getScale()[2])/((Real)1.0 + (Real)2.0*sp[2]/getScale()[2]);
            sp[2]=(Real)0.0;
            return sp;
        }
        else if(isPerspective()==2)
        {
            Coord sp=qrotation.inverseRotate( p-getTranslation() );
            sp[0]=(sp[0]/getScale()[0] + (Real)2.0*sp[2]*camx/getScale()[2])/((Real)1.0 + (Real)2.0*sp[2]/getScale()[2]);
            sp[1]=sp[1]/getScale()[1];
            sp[2]=(Real)0.0;
            return sp;
        }
        else
        {
            Coord sp=qrotation.inverseRotate( p-getTranslation() );
            sp[0]=sp[0]/getScale()[0];
            sp[1]=(sp[1]/getScale()[1] + (Real)2.0*sp[2]*camy/getScale()[2])/((Real)1.0 + (Real)2.0*sp[2]/getScale()[2]);
            sp[2]=(Real)0.0;
            return sp;
        }
    }

    virtual Real toImage(const Real& p) const		{ return (p - getOffsetT())/getScaleT(); }

};

////// infos for Data
class BaseImageTypeInfo
{
public:
    virtual ~BaseImageTypeInfo(){}
};

template<class TDataType>
struct ImageTypeInfo : public BaseImageTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::T BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       }; ///< 1 if this type has valid infos
    enum { FixedSize       = 1                             }; ///< 1 if this type has a fixed size  -> always 1 Image
    enum { ZeroConstructor = 0                             }; ///< 1 if the constructor is equivalent to setting memory to 0  -> I guess so, a default Image is initialzed with nothing
    enum { SimpleCopy      = 0                             }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 0                             }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = 0                             }; ///< 1 if this type uses integer values
    enum { Scalar          = 0                             }; ///< 1 if this type uses scalar values
    enum { Text            = 0                             }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 1                             }; ///< 1 if this type uses copy-on-write -> it seems to be THE important option not to perform too many copies
    enum { Container       = 0                             }; ///< 1 if this type is a container

    enum { Size = 1 }; ///< largest known fixed size for this type, as returned by size()

    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return 1; }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &/*data*/, Index /*index*/, T& /*value*/)
    {
        return;
    }

    template<typename T>
    static void setValue(DataType &/*data*/, Index /*index*/, const T& /*value*/ )
    {
        return;
    }

    static void getValueString(const DataType &data, Index index, std::string& value)
    {
        if (index != 0) return;
        std::ostringstream o; o << data; value = o.str();
    }

    static void setValueString(DataType &data, Index index, const std::string& value )
    {
        if (index != 0) return;
        std::istringstream i(value); i >> data;
    }

    static const void* getValuePtr(const DataType&)
    {
        return nullptr;
    }

    static void* getValuePtr(DataType&)
    {
        return nullptr;
    }
};


template<class T>
struct DataTypeInfo< Image<T> > : public ImageTypeInfo< Image<T> >
{
    static std::string name() { std::ostringstream o; o << "Image<" << DataTypeName<T>::name() << ">"; return o.str(); }
};


}
}

