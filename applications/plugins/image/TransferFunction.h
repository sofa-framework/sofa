/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_IMAGE_TRANSFERFUNCTION_H
#define SOFA_IMAGE_TRANSFERFUNCTION_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

#include <map>

#define LINEAR 0


namespace sofa
{
namespace component
{
namespace engine
{

/**
 * This class transforms pixel intensities
 */

/// Default implementation does not compile
template <class InImageType, class OutImageType>
struct TransferFunctionSpecialization
{
};

/// forward declaration
template <class InImageType, class OutImageType> class TransferFunction;

/// Specialization for regular Image
template <class Ti, class To>
struct TransferFunctionSpecialization<defaulttype::Image<Ti>,defaulttype::Image<To>>
{
    typedef TransferFunction<defaulttype::Image<Ti>,defaulttype::Image<To>> TransferFunctionT;

    static void update(TransferFunctionT& This)
    {
        typename TransferFunctionT::raParam p(This.param);
        typename TransferFunctionT::raImagei in(This.inputImage);
        if(in->isEmpty()) return;
        const cimg_library::CImgList<Ti>& inimg = in->getCImgList();

        typename TransferFunctionT::waImageo out(This.outputImage);
        typename TransferFunctionT::imCoord dim=in->getDimensions();
        out->setDimensions(dim);
        cimg_library::CImgList<To>& img = out->getCImgList();

        switch(This.filter.getValue().getSelectedId())
        {
        case LINEAR:
        {
            typename TransferFunctionT::iomap mp; for(unsigned int i=0; i<p.size(); i+=2) mp[(Ti)p[i]]=(To)p[i+1];
            cimglist_for(inimg,l) cimg_forXYZC(inimg(l),x,y,z,c) img(l)(x,y,z,c)=This.Linear_TransferFunction(inimg(l)(x,y,z,c),mp);
        }
            break;

        default:
            img.assign(in->getCImgList());	// copy
            break;
        }
    }

};


/**
 * \todo adjust type of ParamTypes according to InImageTypes and OutImageTypes
 */
template <class _InImageTypes,class _OutImageTypes>
class TransferFunction : public core::DataEngine
{
    friend struct TransferFunctionSpecialization<_InImageTypes,_OutImageTypes>;

public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE2(TransferFunction,_InImageTypes,_OutImageTypes),Inherited);

    typedef _InImageTypes InImageTypes;
    typedef typename InImageTypes::T Ti;
    typedef typename InImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< InImageTypes > > raImagei;

    typedef _OutImageTypes OutImageTypes;
    typedef typename OutImageTypes::T To;
    typedef helper::WriteOnlyAccessor<Data< OutImageTypes > > waImageo;

    typedef std::map<Ti,To> iomap;
    typedef typename iomap::const_iterator iomapit;


    typedef helper::vector<double> ParamTypes;
    typedef helper::WriteOnlyAccessor<Data< ParamTypes > > waParam;
    typedef helper::ReadAccessor<Data< ParamTypes > > raParam;

    Data<helper::OptionsGroup> filter;
    Data< ParamTypes > param;

    Data< InImageTypes > inputImage;

    Data< OutImageTypes > outputImage;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const TransferFunction<InImageTypes,OutImageTypes>* = NULL) { return InImageTypes::Name()+std::string(",")+OutImageTypes::Name(); }

    TransferFunction()    :   Inherited()
      , filter ( initData ( &filter,"filter","Filter" ) )
      , param ( initData ( &param,"param","Parameters" ) )
      , inputImage(initData(&inputImage,InImageTypes(),"inputImage",""))
      , outputImage(initData(&outputImage,OutImageTypes(),"outputImage",""))
    {
        inputImage.setReadOnly(true);
        outputImage.setReadOnly(true);
        helper::OptionsGroup filterOptions(1	,"0 - Piecewise Linear ( i1, o1, i2, o2 ...)"
                                           );
        filterOptions.setSelectedItem(LINEAR);
        filter.setValue(filterOptions);
    }

    virtual ~TransferFunction() {}

    virtual void init()
    {
        addInput(&inputImage);
        addOutput(&outputImage);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        TransferFunctionSpecialization<InImageTypes,OutImageTypes>::update( *this );
        cleanDirty();
    }


    inline To Linear_TransferFunction(const Ti& vi, const iomap & mp) const
    {
        To vo=mp.begin()->second;
        iomapit mit;
        for (iomapit it=mp.begin(); it!=mp.end(); it++)
        {
            if (it->first>vi && it!=mp.begin())
            {
                double alpha=(((double)it->first-(double)vi)/((double)it->first-(double)mit->first));
                double v= alpha*(double)mit->second + (1.-alpha)*(double)it->second;
                return (To)v;
            }
            else vo=it->second;
            mit=it;
        }
        return vo;
    }


};





} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_IMAGE_TRANSFERFUNCTION_H
