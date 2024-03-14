/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/gl/component/rendering3d/config.h>

#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/vectorLinks.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/gl/component/rendering3d/OglModel.h>

namespace sofa::gl::component::rendering3d
{

/**
 * This class merges several visual models.
 */
class SOFA_GL_COMPONENT_RENDERING3D_API MergeVisualModels : public OglModel
{
public:
    using Inherit = OglModel;
    SOFA_CLASS(MergeVisualModels,Inherit);


    Data<unsigned int> d_nbInput; ///< number of input visual models to merge

    typedef core::objectmodel::SingleLink< MergeVisualModels, VisualModelImpl, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkVisualModel;
    core::objectmodel::VectorLinks< LinkVisualModel, MergeVisualModels > vl_input;


protected:
    MergeVisualModels()
        : d_nbInput( initData (&d_nbInput, (unsigned)1, "nb", "number of input visual models to merge") )
        , vl_input(this,"input", "input visual model")
    {
        vl_input.resize(d_nbInput.getValue());
    }

    ~MergeVisualModels() override {}

    void update()
    {
        unsigned int nb = d_nbInput.getValue();

        size_t nbpos = 0, nbvert = 0, nbedges = 0, nbtris = 0, nbquads = 0, nbrTexC = 0;
        for (unsigned int i=0; i<nb; ++i)
        {
            if (i < vl_input.size())
            {
                if (auto vl_input_i = *vl_input[i])
                {
                    nbpos += vl_input_i->m_positions.getValue().size();
                    nbvert += vl_input_i->m_vertices2.getValue().size();
                    nbrTexC += vl_input_i->m_vtexcoords.getValue().size();
                    nbedges += vl_input_i->m_edges.getValue().size();
                    nbtris += vl_input_i->m_triangles.getValue().size();
                    nbquads += vl_input_i->m_quads.getValue().size();
                }
            }
        }


        VecCoord& pos = *this->m_positions.beginWriteOnly();
        pos.resize( nbpos );

        {
            size_t offset = 0;
            for (unsigned int i=0; i<nb; ++i)
            {
                if (i < vl_input.size())
                {
                    if (auto vl_input_i = *vl_input[i])
                    {
                        const VecCoord& in = vl_input_i->m_positions.getValue();
                        std::copy( in.begin(), in.end(), pos.begin()+offset );
                        offset += in.size();
                    }
                }
            }
            this->m_positions.endEdit();
        }




        {
            VecCoord& vert = *this->m_vertices2.beginWriteOnly();
            vert.resize( nbvert );

            size_t offset = 0;
            for (unsigned int i=0; i<nb; ++i)
            {
                if (i < vl_input.size())
                {
                    if (auto vl_input_i = *vl_input[i])
                    {
                        const VecCoord& in = vl_input_i->m_vertices2.getValue();
                        std::copy( in.begin(), in.end(), vert.begin()+offset );
                        offset += in.size();
                    }
                }
            }
            this->m_vertices2.endEdit();
        }



        {
            auto& vertIdx = *this->m_vertPosIdx.beginWriteOnly();
            vertIdx.resize( nbvert );

            size_t offset = 0;
            size_t offsetIdx = 0;
            for (unsigned int i=0; i<nb; ++i)
            {
                if (i < vl_input.size())
                {
                    if (auto vl_input_i = *vl_input[i])
                    {
                        const auto& in = vl_input_i->m_vertPosIdx.getValue();

                        for( size_t j=0;j<in.size();++j)
                        {
                            auto& e = vertIdx[offset+j];
                            e = in[j];
                            e += offsetIdx;
                        }

                        offset += in.size();
                        offsetIdx += (*vl_input[i])->m_positions.getValue().size();
                    }
                }
            }
            this->m_vertPosIdx.endEdit();
        }

        {
            auto& vertIdx = *this->m_vertNormIdx.beginWriteOnly();
            vertIdx.resize( nbvert );

            size_t offset = 0;
            size_t offsetIdx = 0;
            for (unsigned int i=0; i<nb; ++i)
            {
                if (i < vl_input.size())
                {
                    if (auto vl_input_i = *vl_input[i])
                    {
                        const auto& in = vl_input_i->m_vertNormIdx.getValue();

                        for( size_t j=0;j<in.size();++j)
                        {
                            auto& e = vertIdx[offset+j];
                            e = in[j];
                            e += offsetIdx;
                        }

                        offset += in.size();
                        offsetIdx += vl_input_i->m_positions.getValue().size();
                    }
                }
            }
            this->m_vertNormIdx.endEdit();
        }



        {
            VecTexCoord& vert = *this->m_vtexcoords.beginWriteOnly();
            vert.resize(nbrTexC);

            size_t offset = 0;
            for (unsigned int i=0; i<nb; ++i)
            {
                if (i < vl_input.size())
                {
                    if (auto vl_input_i = *vl_input[i])
                    {
                        const VecTexCoord& in = vl_input_i->m_vtexcoords.getValue();
                        std::copy( in.begin(), in.end(), vert.begin()+offset );
                        offset += in.size();
                    }
                }
            }
            this->m_vtexcoords.endEdit();
        }




        unsigned int offsetPoint = 0;



        Inherit::VecVisualEdge& edges = *this->m_edges.beginWriteOnly();
        edges.resize( nbedges );

        size_t offsetEdge = 0;
        for (unsigned int i=0; i<nb; ++i)
        {
            if (i < vl_input.size())
            {
                if (auto vl_input_i = *vl_input[i])
                {
                    const Inherit::VecVisualEdge& in = vl_input_i->m_edges.getValue();

                    for( size_t j=0;j<in.size();++j)
                    {
                        VisualEdge& e = edges[offsetEdge+j];
                        e = in[j];
                        e[0] += offsetPoint;
                        e[1] += offsetPoint;
                    }

                    offsetEdge += in.size();
                    offsetPoint += (unsigned int)(vl_input_i->m_vertices2.getValue().size());
                }
            }
        }
        this->m_edges.endEdit();





        Inherit::VecVisualTriangle& tris = *this->m_triangles.beginWriteOnly();
        tris.resize( nbtris );

        offsetPoint = 0;
        size_t offsetTri = 0;
        for (unsigned int i=0; i<nb; ++i)
        {
            if (i < vl_input.size())
            {
                if (auto vl_input_i = *vl_input[i])
                {
                    const Inherit::VecVisualTriangle& in = vl_input_i->m_triangles.getValue();

                    for( size_t j=0;j<in.size();++j)
                    {
                        VisualTriangle& t = tris[offsetTri+j];
                        t = in[j];
                        t[0] += offsetPoint;
                        t[1] += offsetPoint;
                        t[2] += offsetPoint;
                    }

                    offsetTri += in.size();
                    offsetPoint += (unsigned int)(vl_input_i->m_vertices2.getValue().size());
                }
            }
        }
        this->m_triangles.endEdit();



        Inherit::VecVisualQuad& quads = *this->m_quads.beginWriteOnly();
        quads.resize( nbquads );

        offsetPoint = 0;
        size_t offsetQuad = 0;
        for (unsigned int i=0; i<nb; ++i)
        {
            if (i < vl_input.size())
            {
                if (auto vl_input_i = *vl_input[i])
                {
                    const Inherit::VecVisualQuad& in = vl_input_i->m_quads.getValue();

                    for( size_t j=0;j<in.size();++j)
                    {
                        VisualQuad& q = quads[offsetQuad+j];
                        q = in[j];
                        q[0] += offsetPoint;
                        q[1] += offsetPoint;
                        q[2] += offsetPoint;
                        q[3] += offsetPoint;
                    }

                    offsetQuad += in.size();
                    offsetPoint += (unsigned int)(vl_input_i->m_vertices2.getValue().size());
                }
            }
        }
        this->m_quads.endEdit();
    }

public:

    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override
    {
        vl_input.parseSizeLinks(arg, d_nbInput);
        Inherit1::parse(arg);
    }
    void parseFields ( const std::map<std::string,std::string*>& str ) override
    {
        vl_input.parseFieldsSizeLinks(str, d_nbInput);
        Inherit1::parseFields(str);
    }

    void init() override
    {
        vl_input.resize(d_nbInput.getValue());
        Inherit1::init();
        update();
    }

    void reinit() override
    {
        vl_input.resize(d_nbInput.getValue());
        Inherit1::reinit();
        update();
    }

};

} // namespace sofa::gl::component::rendering3d
