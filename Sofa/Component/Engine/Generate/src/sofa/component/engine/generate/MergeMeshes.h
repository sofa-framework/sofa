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
#include <sofa/component/engine/generate/config.h>

#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/helper/StringUtils.h>

namespace sofa::component::engine::generate
{

/**
 * This class merge several meshes.
 */
template <class DataTypes>
class MergeMeshes : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MergeMeshes,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef type::vector<unsigned int> VecIndex;

protected:
    MergeMeshes();

    ~MergeMeshes() override;
public:
    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str ) override;

    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data<unsigned int> f_nbMeshes; ///< number of meshes to merge

    type::vector<Data<VecCoord>*> vf_positions;
    type::vector< Data< type::vector< type::fixed_array <unsigned int,2> > >* > vf_edges;
    type::vector< Data< type::vector< type::fixed_array <unsigned int,3> > >* > vf_triangles;
    type::vector< Data< type::vector< type::fixed_array <unsigned int,4> > >* > vf_quads;
    type::vector< Data< type::vector< type::vector<unsigned int> > >* > vf_polygons;
    type::vector< Data< type::vector< type::fixed_array<unsigned int,4> > >* > vf_tetrahedra;
    type::vector< Data< type::vector< type::fixed_array<unsigned int,8> > >* > vf_hexahedra;


    Data<unsigned> f_output_npoints; ///< Number Of out points
    Data<VecCoord> f_output_positions; ///< Output Vertices of the merged mesh
    Data< type::vector< type::fixed_array <unsigned int,2> > > f_output_edges; ///< Output Edges of the merged mesh
    Data< type::vector< type::fixed_array <unsigned int,3> > > f_output_triangles; ///< Output Triangles of the merged mesh
    Data< type::vector< type::fixed_array <unsigned int,4> > > f_output_quads; ///< Output Quads of the merged mesh
    Data< type::vector< type::vector<unsigned int> > > f_output_polygons; ///< Output Polygons of the merged mesh
    Data< type::vector< type::fixed_array<unsigned int,4> > > f_output_tetrahedra; ///< Output Tetrahedra of the merged mesh
    Data< type::vector< type::fixed_array<unsigned int,8> > > f_output_hexahedra; ///< Output Hexahedra of the merged mesh

protected:
    void createInputMeshesData(int nb = -1);
    template<class T>
    void createInputDataVector(unsigned int nb, type::vector< Data<T>* >& vf, std::string name, std::string help)
    {
        vf.reserve(nb);
        for (unsigned int i=vf.size(); i<nb; ++i)
        {
            std::ostringstream oname, ohelp;
            oname << name << (i+1);
            ohelp << help << (i+1);
            std::string name_i = oname.str();
            std::string help_i = ohelp.str();
            Data<T>* d = new Data<T>(sofa::helper::getAStringCopy(help_i.c_str()), true, false);
            d->setName(name_i);
            vf.push_back(d);
            this->addData(d);
            this->addInput(d);
        }
    }
    template<class T>
    void deleteInputDataVector(type::vector< Data<T>* >& vf)
    {
        for (unsigned int i=0; i<vf.size(); ++i)
        {
            this->delInput(vf[i]);
            delete vf[i];
        }
        vf.clear();
    }
    template<class T>
    void mergeInputDataVector(unsigned int nb, Data<T>& outF, const type::vector< Data<T>* >& inVF, const type::vector< Data<VecCoord>* >& inVFPos)
    {
        unsigned int nelems = 0;
        for (unsigned int i=0; i<nb; ++i)
            nelems += inVF[i]->getValue().size();
        helper::WriteOnlyAccessor< Data<T> > out = outF;
        out.clear();
        out.reserve(nelems);
        unsigned int shift = 0;
        for (unsigned int i=0; i<nb; ++i)
        {
            helper::ReadAccessor< Data<T> > in = inVF[i];
            for (unsigned int j=0; j<in.size(); ++j)
            {
                typename T::value_type outT = in[j];
                for (unsigned int k=0; k<outT.size(); ++k)
                    outT[k] += shift;
                out.push_back(outT);
            }
            shift += inVFPos[i]->getValue().size();
        }
    }
    void mergeInputDataVector(unsigned int nb, Data<VecCoord>& outF, const type::vector< Data<VecCoord>* >& inVF)
    {
        unsigned int nelems = 0;
        for (unsigned int i=0; i<nb; ++i)
            nelems += inVF[i]->getValue().size();
        helper::WriteOnlyAccessor< Data<VecCoord> > out = outF;
        out.clear();
        out.reserve(nelems);
        for (unsigned int i=0; i<nb; ++i)
        {
            helper::ReadAccessor< Data<VecCoord> > in = inVF[i];
            for (unsigned int j=0; j<in.size(); ++j)
                out.push_back(in[j]);
        }
    }
};

#if !defined(SOFA_COMPONENT_ENGINE_MERGEMESHES_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeMeshes<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeMeshes<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeMeshes<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeMeshes<defaulttype::Rigid2Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergeMeshes<defaulttype::Rigid3Types>;
 
#endif

} //namespace sofa::component::engine::generate
