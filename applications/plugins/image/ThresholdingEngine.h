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
 * This class computes an automatic image threshold that would separate foreground from background
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


    enum {BHT/*Balanced histogram thresholding*/=0, OTSU /*Otsu's method*/,  NB_METHODS};
    Data<helper::OptionsGroup> d_method;
    Data< ParamTypes >  d_param;

    Data<  ImageTypes >  d_image;

    Data< Ti > d_threshold;

     ThresholdingEngine()
      : Inherit1()
      , d_method ( initData ( &d_method,"method","Thresholding method" ) )
      , d_param ( initData ( & d_param,"param","Parameters" ) )
      , d_image(initData(&d_image, ImageTypes(),"image","input image"))
      , d_threshold(initData(&d_threshold, Ti(),"threshold","Computed threshold"))
    {
        d_image.setReadOnly(true);
        helper::OptionsGroup methodOptions(2
                                           , "0 - BHT ( histogramSize )"
                                           , "1 - Otsu ( histogramSize )"
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

        Ti& threshold = *d_threshold.beginWriteOnly();

        if(image.isEmpty()) { threshold = Ti(); return; }


        switch(d_method.getValue().getSelectedId())
        {
        case BHT:
        {
            unsigned histogramSize = param.empty() ? 256 : param[0];

            Ti tmin,tmax;
            threshold = BHThreshold( image.get_histogram( tmin, tmax, histogramSize, true ) );
            break;
        }

        case OTSU:
        {
            unsigned histogramSize = param.empty() ? 256 : param[0];

            Ti tmin,tmax;
            const cimg_library::CImg<Ti> img = image.getCImgList()(0);
            threshold = OtsuThreshold( image.get_histogram( tmin, tmax, histogramSize, true ), image.getCImgList().size()*img.width()*img.height()*img.depth() );
            break;
        }

        default:
            assert(false);
            break;
        }

        d_threshold.endEdit();

    }

public:

     virtual std::string getTemplateName() const { return templateName(this);    }
     static std::string templateName(const  ThresholdingEngine< ImageTypes>* = NULL) { return  ImageTypes::Name(); }


protected:


     // TODO move these generic method where they can be used by someoneelse


//https://en.wikipedia.org/wiki/Balanced_histogram_thresholding
     static Ti BHThreshold( const cimg_library::CImg<unsigned>& /*histogram*/)
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
     static Ti OtsuThreshold( const cimg_library::CImg<unsigned>& histogram, size_t totalImageSize )
     {
         double sum = 0;

         cimg_foroff( histogram, i )
             sum += i * histogram[i];

         double sumB = 0;
         double wB = 0;
         double wF = 0;

         double max = 0;
         Ti threshold = 0;

         cimg_foroff( histogram, i )
         {
             wB += histogram[i];
             if (wB == 0) continue;

             wF = totalImageSize - wB;
             if (wF == 0) continue;

             sumB += (double)(i * histogram[i]);

             double mB = sumB / wB;
             double mF = (sum - sumB) / wF;

             double between = wB * wF * (mB - mF) * (mB - mF);

             if (between > max)
             {
                 max = between;
                 threshold = histogram[i];
             }
         }
         return threshold;
     }


};


} // namespace engine

} // namespace component

} // namespace sofa


#endif
