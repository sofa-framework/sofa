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
#ifndef FLEXIBLE_ImageShapeFunctionSelectNode_H
#define FLEXIBLE_ImageShapeFunctionSelectNode_H

#include <Flexible/config.h>
#include <sofa/core/DataEngine.h>

#include "BaseImageShapeFunction.h"

namespace sofa
{
namespace component
{
namespace engine
{

/**
 * From an image shape function, this engine builds an image of the weights of a given node.
 *
 * @author Thomas Lemaire
 * @date   2014
 */
template <class TImageTypes>
class ImageShapeFunctionSelectNode : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ImageShapeFunctionSelectNode, TImageTypes) , core::DataEngine);

    typedef TImageTypes ImageTypes;

    /** @name shape function */
    //@{

    typedef typename ImageTypes::imCoord imCoord;
    typedef typename sofa::component::shapefunction::BaseImageShapeFunctionSpecialization<ImageTypes>::DistT DistT;
    typedef typename sofa::component::shapefunction::BaseImageShapeFunctionSpecialization<ImageTypes>::DistTypes DistTypes;
    typedef helper::ReadAccessor<Data< DistTypes > > raDist;
    Data< DistTypes > d_weights; ///< weights of the shape function

    typedef typename sofa::component::shapefunction::BaseImageShapeFunctionSpecialization<ImageTypes>::IndT IndT;
    typedef typename sofa::component::shapefunction::BaseImageShapeFunctionSpecialization<ImageTypes>::IndTypes IndTypes;
    typedef helper::ReadAccessor<Data< IndTypes > > raInd;
    Data< IndTypes > d_indices; ///< indices of the shape function

    Data<unsigned int> d_nodeIndex; ///< index of the selected node

    typedef helper::WriteOnlyAccessor<Data< DistTypes > > waDist;
    Data< DistTypes > d_nodeWeights; ///< weights of the selected node
    //@}

    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName(const ImageShapeFunctionSelectNode<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    ImageShapeFunctionSelectNode()
        : d_weights(initData(&d_weights,DistTypes(),"shapeFunctionWeights","shapeFunction weights image"))
        , d_indices(initData(&d_indices,IndTypes(),"shapeFunctionIndices","shapeFunction indices image"))
        , d_nodeIndex(initData(&d_nodeIndex,"nodeIndex","index of parent node to select"))
        , d_nodeWeights(initData(&d_nodeWeights,DistTypes(),"nodeWeights","selected node weights image"))

    {}

    virtual void init() {
        addInput(&d_weights);
        addInput(&d_indices);
        addInput(&d_nodeIndex);
        addOutput(&d_nodeWeights);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        unsigned int nodeIndex = d_nodeIndex.getValue();
        sout << "Update image for node " << nodeIndex << sendl;
        raDist weightData(this->d_weights);
        cimg_library::CImg<DistT> const& weight=weightData->getCImg();
        raInd indicesData(this->d_indices);
        cimg_library::CImg<IndT> const& indices=indicesData->getCImg();

        waDist nodeWeighData(this->d_nodeWeights);
        imCoord dim = weightData->getDimensions();
        // only one node
        dim[ImageTypes::DIMENSION_S]=1;
        nodeWeighData->setDimensions(dim);
        cimg_library::CImg<DistT>& nodeWeigh = nodeWeighData->getCImg();
        nodeWeigh.fill(0);

        // loop over index image
        for(int z=0; z<indices.depth(); z++)
            for(int y=0; y<indices.height(); y++)
                for(int x=0; x<indices.width(); x++)
                    for(int i=0; i<indices.spectrum(); i++)
                        if (indices(x,y,z,i)==d_nodeIndex.getValue())
                            nodeWeigh(x,y,z,0)=weight(x,y,z,i);

        cleanDirty();
    }
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // FLEXIBLE_ImageShapeFunctionSelectNode_H
