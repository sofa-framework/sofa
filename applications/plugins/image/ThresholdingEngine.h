#ifndef SOFA_IMAGE_THRESHOLDING_H
#define SOFA_IMAGE_THRESHOLDING_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/OptionsGroup.h>


namespace sofa
{

namespace component
{

namespace engine
{





/**
 * This class computes an automatic image threshold that would separate foreground from background.

 @author Matthieu Nesme, Benjamin Gilles
 @date 2016

 */


template <class  _ImageTypes>
class ThresholdingEngine : public core::DataEngine
{
public:
    SOFA_CLASS( SOFA_TEMPLATE( ThresholdingEngine, _ImageTypes ), core::DataEngine );

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T Ti;
    typedef typename ImageTypes::imCoord imCoordi;
    typedef SReal Real;

    typedef helper::vector<double> ParamTypes;


    enum {BHT/*Balanced histogram thresholding*/=0, OTSU /*Otsu's method*/,  GMM /*Gaussian Mixture fit*/, NB_METHODS};
    Data<helper::OptionsGroup> d_method;
    Data< ParamTypes >  d_param;

    Data<  ImageTypes >  d_image;

    Data< helper::vector<Ti> > d_threshold;

    ThresholdingEngine()
        : Inherit1()
        , d_method ( initData ( &d_method,"method","Thresholding method" ) )
        , d_param ( initData ( & d_param,"param","Parameters" ) )
        , d_image(initData(&d_image, ImageTypes(),"image","input image"))
        , d_threshold(initData(&d_threshold,"threshold","Computed threshold(s)"))
    {
        d_image.setReadOnly(true);
        helper::OptionsGroup methodOptions(NB_METHODS
                                           , "0 - BHT ( histogramSize )"
                                           , "1 - Otsu ( histogramSize )"
                                           , "2 - Gaussian Mixture Model ( histogramSize, numberOfClasses )"
                                           );
        methodOptions.setSelectedItem(BHT);
        d_method.setValue(methodOptions);
    }

    virtual ~ThresholdingEngine() {}

    virtual void init()
    {
        addInput(&d_image);
        addInput(&d_method);
        addInput(&d_param);
        addOutput(&d_threshold);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:



    virtual void update()
    {
        const helper::vector<double>& param = d_param.getValue();
        const ImageTypes& image = this->d_image.getValue();
        d_method.updateIfDirty();

        cleanDirty();

        if(image.isEmpty()) return;


        unsigned histogramSize = param.empty() ? (image.getCImgList().max()-image.getCImgList().min()+1) : param[0];

        //        sout << "histogramSize: " << histogramSize << sendl;

        Ti tmin, tmax;

        // TODO pass the histogram by reference to save a copy
//        cimg_library::CImg<unsigned int> histogram = image.get_histogram( tmin, tmax, histogramSize, true );
                cimg_library::CImg<unsigned int> histogram = get_normalizedHistogram( tmin, tmax, histogramSize, image.getCImgList() );

        helper::vector<unsigned> thresholds;
        switch(d_method.getValue().getSelectedId())
        {
        case BHT:
        {
            thresholds.push_back( BHThreshold( histogram ) );
            break;
        }

        case OTSU:
        {
            thresholds.push_back( OtsuThreshold( histogram ) );
            break;
        }

        case GMM:
        {
            const unsigned nbClasses = param.size()>1 ? param[1]:2;
            GMMThresholds(thresholds,histogram, nbClasses );
            break;
        }

        default:
            assert(false);
            break;
        }

        // get threshold in Ti range from histogram range
        Real fact = ((Real)tmax-(Real)tmin) / ((Real)histogram.width()-1.);

        helper::vector<Ti>& output = *d_threshold.beginWriteOnly();
        output.clear();
        for(unsigned i=0;i<thresholds.size();++i)  output.push_back( (Ti) (fact * (Real)thresholds[i] + (Real)tmin) );
        d_threshold.endEdit();
    }

public:

    virtual std::string getTemplateName() const { return templateName(this);    }
    static std::string templateName(const  ThresholdingEngine< ImageTypes>* = NULL) { return  ImageTypes::Name(); }


protected:


    // TODO move these generic method where they can be used by someoneelse


    //https://en.wikipedia.org/wiki/Balanced_histogram_thresholding
    static unsigned BHThreshold( const cimg_library::CImg<unsigned>& /*histogram*/)
    {
        msg_warning("ThresholdingEngine") << "BHThreshold not yet implemented";

        //         Ti i_m = (Ti)((i_s + i_e) / 2.0f); // center of the weighing scale I_m
        //         w_l = get_weight(i_s, i_m + 1, histogram); // weight on the left W_l
        //         w_r = get_weight(i_m + 1, i_e + 1, histogram); // weight on the right W_r
        //         while (i_s <= i_e) {
        //             if (w_r > w_l) { // right side is heavier
        //                 w_r -= histogram[i_e--];
        //                 if (((i_s + i_e) / 2) < i_m) {
        //                     w_r += histogram[i_m];
        //                     w_l -= histogram[i_m--];
        //                 }
        //             } else if (w_l >= w_r) { // left side is heavier
        //                 w_l -= histogram[i_s++];
        //                 if (((i_s + i_e) / 2) >= i_m) {
        //                     w_l += histogram[i_m + 1];
        //                     w_r -= histogram[i_m + 1];
        //                     i_m++;
        //                 }
        //             }
        //         }
        //         return i_m;
        return Ti();
    }

    // https://en.wikipedia.org/wiki/Otsu%27s_method
    static unsigned OtsuThreshold( const cimg_library::CImg<unsigned>& histogram)
    {
        size_t totalImageSize = histogram.sum();

        double sum = 0;

        cimg_foroff( histogram, i )
                sum += i * histogram[i];

        double sumB = 0;
        double wB = 0;
        double wF = 0;

        double max = 0;
        unsigned threshold = 0;

        cimg_foroff( histogram, i )
        {
            wB += histogram[i];
            if (wB == 0) continue;

            wF = totalImageSize - wB;
            if (wF == 0) continue;

            sumB += i * histogram[i];

            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;

            double between = wB * wF * (mB - mF) * (mB - mF);

            if( between > max )
            {
                threshold = i;
                max = between;
            }
        }
        return threshold;
    }


    // gaussian mixture model using EM algorithm
    // http://www.cs.duke.edu/courses/spring04/cps196.1/handouts/EM/tomasiEM.pdf
    static inline Real Gaussian(const Real v,const Real m,const Real s)     { return exp(-(v-m)*(v-m)/(2.*s*s))/(sqrt(2.*cimg_library::cimg::PI)*s);    }
    static inline Real GaussianMix(const Real v,const std::vector<Real> &p,const std::vector<Real> &m,const std::vector<Real> &s)     {  Real ret=0; for(unsigned int i=0;i<p.size();i++) ret+=p[i]*Gaussian(v,m[i],s[i]); return ret;    }
    static inline Real GaussianCond(const unsigned int i,const Real v,const std::vector<Real>& p,const std::vector<Real> &m,const std::vector<Real> &s)    {  Real mix=GaussianMix(v,p,m,s); if(mix!=0) return p[i]*Gaussian(v,m[i],s[i])/mix; else return (Real)0.;  }
    static inline std::pair<Real,unsigned> MaxGaussian(const Real v,const std::vector<Real>& p,const std::vector<Real> &m,const std::vector<Real> &s)
    {
        std::pair<Real,unsigned> maxipair(p[0]*Gaussian(v,m[0],s[0]),0);
        for(unsigned int i=1;i<p.size();i++) { Real val=p[i]*Gaussian(v,m[i],s[i]); if(val>maxipair.first) {maxipair.first=val; maxipair.second=i;} }
        return maxipair;
    }

    static void GMMThresholds(helper::vector<unsigned>& thresholds, const cimg_library::CImg<unsigned>& histogram, const size_t nbClasses )
    {
        const double Tol=1E-2;
        const unsigned int Itmax=200;

        unsigned int numBins=histogram.width(),N=histogram.sum(), count=0;

        // compute GMM
        std::vector<Real> p(nbClasses),m(nbClasses),s(nbClasses); // gaussian amplitude, mean and sigma
        std::vector<Real> newp(nbClasses),newm(nbClasses),news(nbClasses);

        Real step = (Real)numBins/((Real)nbClasses-1.);
        for(unsigned int i=0;i<nbClasses;i++)  {p[i]=1./(Real)nbClasses; m[i]=i*step; s[i]=step; } // init

        bool stop=false;
        while(!stop)
        {
            // E step
            for(unsigned i=0;i<nbClasses;i++) { newp[i]=0; for(unsigned j=0;j<numBins;j++) newp[i]+=(Real)histogram[j]*GaussianCond(i,j,p,m,s); }
            // M step
            for(unsigned i=0;i<nbClasses;i++) { newm[i]=0; for(unsigned j=0;j<numBins;j++) newm[i]+=(Real)histogram[j]*((Real)j)*GaussianCond(i,j,p,m,s);  newm[i]/=newp[i]; }
            for(unsigned i=0;i<nbClasses;i++) { news[i]=0; for(unsigned j=0;j<numBins;j++) news[i]+=(Real)histogram[j]*((Real)j-newm[i])*((Real)j-newm[i])*GaussianCond(i,j,p,m,s); news[i]=sqrt(news[i]/newp[i]); }

            double var=0; for(unsigned i=0;i<nbClasses;i++) var+=(newm[i]-m[i])*(newm[i]-m[i]);
            for(unsigned i=0;i<nbClasses;i++)
            {
                p[i]=newp[i]/(Real)N;
                m[i]=newm[i];
                s[i]=news[i];
            }
            count++;
            if(var<Tol || count>=Itmax) stop=true;
        }

        // find thresholds from GMM
        count=0;
        std::pair<Real,unsigned> oldmaxipair = MaxGaussian(0,p,m,s);
        for(unsigned j=1;j<numBins-1;j++)
        {
            std::pair<Real,unsigned> maxipair = MaxGaussian(j,p,m,s);
            if (oldmaxipair.second!=maxipair.second)
            {
                oldmaxipair=maxipair;
                thresholds.push_back(j);
                count+=1;
            }
            if(count==nbClasses-1)
                break;
        }
    }



    cimg_library::CImg<unsigned int> get_normalizedHistogram(Ti& value_min, Ti& value_max, const unsigned int histogramSize, const cimg_library::CImgList<Ti>& img)
    {
        value_min=img.min();
        value_max=img.max();
        if(value_min==value_max) value_max=value_min+(Ti)1;
        unsigned int nbvalues = value_max-value_min+1; // range for all possible number of integer values
        if(nbvalues<1024) nbvalues=1024; // for float images, ensure large enough range

        cimg_library::CImg<unsigned int> histogram(nbvalues,1,1,1,0);
        long double mul= ((long double)nbvalues-1.)/((long double)value_max-(long double)value_min);
        cimglist_for(img,l) cimg_foroff(img(l),off)
        {
            long double v=((long double)img(l)[off]-(long double)value_min)*mul;
            if(v<0) v=0; else if(v>(long double)(nbvalues-1)) v=(long double)(nbvalues-1);
            ++histogram((int)(v));
        }
        // normalize to have 'histogramSize' bins with similar numbers of samples
        unsigned int count=0,j=0,nbmax=histogram.sum()/histogramSize;
        for(unsigned int i=0;i<nbvalues;i++)
        {
            count+=histogram[i];
            if(count>=nbmax)
            {
                for(unsigned int bin=j;bin<=i;bin++) histogram[bin]=(double)count/((double)i-(double)j+1.);
                count=0; j=i+1;
            }
        }
        if(count!=0) for(unsigned int bin=j;bin<nbvalues;bin++) histogram[bin]=(double)count/((double)nbvalues-(double)j);
        return histogram;
    }








};


} // namespace engine

} // namespace component

} // namespace sofa


#endif
