/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef CGALPLUGIN_MESHGENERATIONFROMIMAGE_INL
#define CGALPLUGIN_MESHGENERATIONFROMIMAGE_INL
#include "MeshGenerationFromImage.h"
#include <sofa/defaulttype/Quat.h>

using namespace sofa;

#define SQR(X)   ((X)*(X))

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,5,0)
using namespace CGAL::parameters;
#endif

namespace cgal
{

template <class DataTypes, class _ImageTypes>
MeshGenerationFromImage<DataTypes, _ImageTypes>::MeshGenerationFromImage()
    : d_filename(initData(&d_filename,"filename","Image file"))
    , d_image(initData(&d_image,ImageTypes(),"image","image input"))
    , d_transform(initData(&d_transform, "transform" , "12-param vector for trans, rot, scale, ..."))
    , d_features( initData (&d_features, "features", "features (1D) that will be preserved in the mesh"))
    , d_newX0( initData (&d_newX0, "outputPoints", "New Rest position coordinates from the tetrahedral generation"))
    , d_tetrahedra(initData(&d_tetrahedra, "outputTetras", "List of tetrahedra"))
    , d_tetraDomain(initData(&d_tetraDomain, "outputTetrasDomains", "domain of each tetrahedron"))
    , d_outputCellData(initData(&d_outputCellData, "outputCellData", "Output cell data"))
    , d_frozen(initData(&d_frozen, false, "frozen", "true to prohibit recomputations of the mesh"))
    , d_edgeSize(initData(&d_edgeSize, 2.0, "edgeSize", "Edge size criterium (needed for polyline features"))
    , d_facetAngle(initData(&d_facetAngle, 25.0, "facetAngle", "Lower bound for the angle in degrees of the surface mesh facets"))
    , d_facetSize(initData(&d_facetSize, 0.15, "facetSize", "Uniform upper bound for the radius of the surface Delaunay balls"))
    , d_facetApproximation(initData(&d_facetApproximation, 0.008, "facetApproximation", "Upper bound for the center-center distances of the surface mesh facets"))
    , d_cellRatio(initData(&d_cellRatio, 4.0, "cellRatio", "Upper bound for the radius-edge ratio of the tetrahedra"))
    , d_cellSize(initData(&d_cellSize, 1.0, "cellSize", "Uniform upper bound for the circumradii of the tetrahedra in the mesh"))
    , d_label(initData(&d_label, "label", "label to be resized to a specific cellSize"))
    , d_labelCellSize(initData(&d_labelCellSize, "labelCellSize", "Uniform upper bound for the circumradii of the tetrahedra in the mesh by label"))
    , d_labelCellData(initData(&d_labelCellData, "labelCellData", "1D cell data by label"))
    , d_odt(initData(&d_odt, false, "odt", "activate odt optimization"))
    , d_lloyd(initData(&d_lloyd, false, "lloyd", "activate lloyd optimization"))
    , d_perturb(initData(&d_perturb, false, "perturb", "activate perturb optimization"))
    , d_exude(initData(&d_exude, false, "exude", "activate exude optimization"))
    , d_odtMaxIt(initData(&d_odtMaxIt, 200, "odt_max_it", "odt max iteration number"))
    , d_lloydMaxIt(initData(&d_lloydMaxIt, 200, "lloyd_max_it", "lloyd max iteration number"))
    , d_perturbMaxTime(initData(&d_perturbMaxTime, 20.0, "perturb_max_time", "perturb maxtime"))
    , d_exudeMaxTime(initData(&d_exudeMaxTime, 20.0, "exude_max_time", "exude max time"))
    , d_ordering(initData(&d_ordering, 0, "ordering", "Output points and elements ordering (0 = none, 1 = longest bbox axis)"))
    , d_drawTetras(initData(&d_drawTetras, false, "drawTetras", "display generated tetra mesh"))
{
}

template<class T1, class T2> bool compare_pair_first(const std::pair<T1,T2>& e1, const std::pair<T1,T2>& e2)
{
    return e1.first < e2.first;
}

template <class DataTypes, class _ImageTypes>
void MeshGenerationFromImage<DataTypes, _ImageTypes>::init()
{
    addOutput(&d_newX0);
    addOutput(&d_tetrahedra);
    addOutput(&d_tetraDomain);
    addOutput(&d_outputCellData);
    addInput(&d_frozen);
    addInput(&d_facetAngle);
    addInput(&d_facetSize);
    addInput(&d_facetApproximation);
    addInput(&d_cellRatio);
    addInput(&d_cellSize);
    addInput(&d_label);
    addInput(&d_labelCellSize);
    addInput(&d_odt);
    addInput(&d_lloyd);
    addInput(&d_perturb);
    addInput(&d_exude);
    addInput(&d_odtMaxIt);
    addInput(&d_lloydMaxIt);
    addInput(&d_perturbMaxTime);
    addInput(&d_exudeMaxTime);
    addInput(&d_ordering);
    addInput(&d_image);
    addInput(&d_transform);

    setDirtyValue();
}

template <class DataTypes, class _ImageTypes>
void MeshGenerationFromImage<DataTypes, _ImageTypes>::reinit()
{
    sofa::core::DataEngine::reinit();
    update();
}

template <class C3t3>
int countWellCentered(C3t3& c3t3)
{
    int nb_in = 0;
    const typename C3t3::Triangulation& tri = c3t3.triangulation();
    for (typename C3t3::Cell_iterator cit = c3t3.cells_begin(); cit != c3t3.cells_end(); ++cit )
    {
        if (K().has_on_bounded_side_3_object()(tri.tetrahedron(cit),tri.dual(cit)))
        {
            ++nb_in;
        }
    }
    return nb_in;
}

template <class C3t3,class Obj>
void printStats(C3t3& c3t3, Obj* obj, const char* step = "")
{
    int nb_in = countWellCentered(c3t3);

    msg_info(obj) << step << ":  number of tetra     = " << c3t3.number_of_cells();
    msg_info(obj) << step << ":  well-centered tetra = " << ((double)nb_in/(double)c3t3.number_of_cells())*100 << "%";
}

template <class DataTypes, class _ImageTypes>
void MeshGenerationFromImage<DataTypes, _ImageTypes>::update()
{
    helper::WriteAccessor< Data<VecCoord> > newPoints = d_newX0;
    helper::WriteAccessor< Data<SeqTetrahedra> > tetrahedra = d_tetrahedra;
    helper::WriteAccessor<Data< vector<int> > > tetraDomain(this->d_tetraDomain);

    helper::ReadAccessor< Data<VecCoord> > fts = d_features;


    if (d_frozen.getValue())
        return;

    newPoints.clear();
    tetrahedra.clear();
    tetraDomain.clear();

    // Create domain
    msg_info(this) << "Create domain";
    CGAL::Image_3 image3;

    if (this->d_filename.getFullPath().empty()) {
        if(d_image.isSet()) {
            raImage in(this->d_image);
            raTransform inT(this->d_transform);
            unsigned int i,j,k;
            uint8_t *volumearray =new uint8_t[in->getDimensions()[2]*in->getDimensions()[1]*in->getDimensions()[0]];
            uint8_t *vptr= volumearray;
            for(k=0;k<in->getDimensions()[2];k++){
                for(j=0;j<in->getDimensions()[1];j++){
                    for(i=0;i<in->getDimensions()[0];i++){
                        vptr[k*in->getDimensions()[0]*in->getDimensions()[1]+j*in->getDimensions()[0]+i] =
                                in->getCImg()[k*in->getDimensions()[0]*in->getDimensions()[1]+j*in->getDimensions()[0]+i]; //get_vector_at(x,y,z)
                    }
                }
            }
            _image* vrnimage = _initImage();
            vrnimage->vectMode = VM_SCALAR;
            //image dimension
            vrnimage->xdim = in->getDimensions()[0]; //columns
            vrnimage->ydim = in->getDimensions()[1]; //rows
            vrnimage->zdim = in->getDimensions()[2]; //planes
            vrnimage->vdim = 1; //vectorial dimension
            //voxel size
            vrnimage->vx = inT->getScale()[0];
            vrnimage->vy = inT->getScale()[1];
            vrnimage->vz = inT->getScale()[2];
            //image translation
            vrnimage->tx = inT->getTranslation()[0];
            vrnimage->ty = inT->getTranslation()[1];
            vrnimage->tz = inT->getTranslation()[2];
            //image rotation
            vrnimage->rx = inT->getRotation()[0];
            vrnimage->ry = inT->getRotation()[1];
            vrnimage->rz = inT->getRotation()[2];
            vrnimage->endianness = END_LITTLE;
            vrnimage->wdim = 1;
            vrnimage->wordKind = WK_FIXED;
            vrnimage->sign = SGN_UNSIGNED;
            vrnimage->data = ImageIO_alloc(in->getDimensions()[2]*in->getDimensions()[1]*in->getDimensions()[0]);
            memcpy(vrnimage->data,(void*)volumearray,in->getDimensions()[2]*in->getDimensions()[1]*in->getDimensions()[0]);
            image3 = CGAL::Image_3(vrnimage);
        }
        else
        {
            msg_error(this) << "ERROR : image filename is empty";
            return;
        }
    } else {
        image3.read(this->d_filename.getFullPath().c_str());
    }

    Mesh_domain domain(image3);

    int volume_dimension = 3;
    Sizing_field size(d_cellSize.getValue());

    if (d_label.getValue().size() == d_labelCellSize.getValue().size())
	{
        for (unsigned int i=0; i<d_label.getValue().size(); ++i)
		{
            size.set_size(d_labelCellSize.getValue()[i], volume_dimension,
                domain.index_from_subdomain_index(d_label.getValue()[i]));
		}
	}
	else
	{
        msg_error(this) << "ERROR : label and labelCellSize must have the same size... otherwise cellSize "
            << d_cellSize.getValue() << " will be apply for all layers";
	}


#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,6,0)
    msg_info(this) << "Create Mesh";
    Mesh_criteria criteria(edge_size=d_edgeSize.getValue(),
        facet_angle=d_facetAngle.getValue(), facet_size=d_facetSize.getValue(), facet_distance=d_facetApproximation.getValue(),
        cell_radius_edge=d_cellRatio.getValue(), cell_size=size);

    size_t nfts = fts.size();
    Polylines polylines (nfts);

    if (nfts > 0)
        msg_info(this) << "Explicitly defined 0D features: #" << nfts;

    size_t fi = 0;

    Vector3 translation = Vector3(image3.image()->tx, image3.image()->ty, image3.image()->tz);
    for (Polylines::iterator l = polylines.begin(); l != polylines.end(); l++, fi++) {
        Coord ft = fts[fi];

        ft = ft - translation;

        Point3 p=Point3(ft[0],ft[1], ft[2]);
        l->push_back(p);
        Point3 p1=Point3(ft[0], ft[1], ft[2]);
        l->push_back(p1);
    }
    domain.add_features(polylines.begin(), polylines.end());

    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude());


#else
    // Set mesh criteria
    Facet_criteria facet_criteria(d_facetAngle.getValue(), d_facetSize.getValue(), d_facetApproximation.getValue()); // angle, size, approximation
    Cell_criteria cell_criteria(d_cellRatio.getValue(), d_cellSize.getValue()); // radius-edge ratio, size
    Mesh_criteria criteria(facet_criteria, cell_criteria);

    msg_info(this) << "Create Mesh";
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude());
#endif // CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,6,0)
    printStats(c3t3,this,"Initial mesh");
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,5,0)
    msg_info(this) << "Optimize Mesh";
    if(d_lloyd.getValue())
    {
        CGAL::lloyd_optimize_mesh_3(c3t3, domain, max_iteration_number=d_lloydMaxIt.getValue());
        printStats(c3t3,this,"Lloyd");
    }
    if(d_odt.getValue())
    {
        CGAL::odt_optimize_mesh_3(c3t3, domain, max_iteration_number=d_odtMaxIt.getValue());
        printStats(c3t3,this,"ODT");
    }
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,6,0)
    if(d_perturb.getValue())
    {
        CGAL::perturb_mesh_3(c3t3, domain, time_limit=d_perturbMaxTime.getValue());
        printStats(c3t3,this,"Perturb");
    }
    if(d_exude.getValue())
    {
        CGAL::exude_mesh_3(c3t3, time_limit=d_exudeMaxTime.getValue());
        printStats(c3t3,this,"Exude");
    }
#else
    if(d_perturb.getValue())
    {
        CGAL::perturb_mesh_3(c3t3, domain, max_time=d_perturbMaxTime.getValue());
        printStats(c3t3,this,"Perturb");
    }
    if(d_exude.getValue())
    {
        CGAL::exude_mesh_3(c3t3, max_time=d_exudeMaxTime.getValue());
        printStats(c3t3,this,"Exude");
    }
#endif // CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,6,0)
#endif // CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,5,0)


    const Tr& tr = c3t3.triangulation();
    std::map<Vertex_handle, int> Vnbe;

    for( Cell_iterator cit = c3t3.cells_begin() ; cit != c3t3.cells_end() ; ++cit )
    {
        for (int i=0; i<4; i++)
            ++Vnbe[cit->vertex(i)];
    }
    std::map<Vertex_handle, int> V;
    newPoints.clear();
    int inum = 0;
    int notconnected = 0;
    Point bbmin, bbmax;
    for( Finite_vertices_iterator vit = tr.finite_vertices_begin(); vit != tr.finite_vertices_end(); ++vit)
    {
        Point_3 pointCgal = vit->point();
        Point p;
        p[0] = CGAL::to_double(pointCgal.x());
        p[1] = CGAL::to_double(pointCgal.y());
        p[2] = CGAL::to_double(pointCgal.z());
        if (Vnbe.find(vit) == Vnbe.end() || Vnbe[vit] <= 0)
        {
            nmsg_info(this) << "Un-connected point: " << p;;
            ++notconnected;
        }
        else
        {
            V[vit] = inum++;
            if (newPoints.empty())
                bbmin = bbmax = p;
            else
                for (unsigned int c=0; c<p.size(); c++)
                    if (p[c] < bbmin[c]) bbmin[c] = p[c]; else if (p[c] > bbmax[c]) bbmax[c] = p[c];

            Vector3 rotation = Vector3(image3.image()->rx, image3.image()->ry, image3.image()->rz);
            defaulttype::Quaternion q = helper::Quater<Real>::createQuaterFromEuler(rotation*M_PI/180.0);
            p= q.rotate(p);

            Vector3 translation = Vector3(image3.image()->tx, image3.image()->ty, image3.image()->tz);
            newPoints.push_back(p+translation);
        }
    }
    if (notconnected > 0)
        msg_info(this) << notconnected << " points are not connected to the mesh.";

    tetrahedra.clear();
    tetraDomain.clear();
    m_tetraDomainLabels.clear();
    for( Cell_iterator cit = c3t3.cells_begin() ; cit != c3t3.cells_end() ; ++cit )
    {
        Tetra tetra;

        for (int i=0; i<4; i++)
            tetra[i] = V[cit->vertex(i)];
        tetrahedra.push_back(tetra);
        if( std::find(m_tetraDomainLabels.begin(),m_tetraDomainLabels.end(),c3t3.subdomain_index(cit)) == m_tetraDomainLabels.end()) {
            m_tetraDomainLabels.push_back(c3t3.subdomain_index(cit));
        }
        tetraDomain.push_back(c3t3.subdomain_index(cit));
    }

    int nbp = newPoints.size();
    int nbe = tetrahedra.size();

    switch(d_ordering.getValue())
    {
    case 0: break;
    case 1:
    {
        int axis = 0;
        for (int c=1; c<3; c++)
            if (bbmax[c]-bbmin[c] > bbmax[axis]-bbmin[axis]) axis=c;
        msg_info(this) << "Ordering along the " << (char)('X'+axis) << " axis.";
        helper::vector< std::pair<float,int> > sortArray;
        for (int i=0; i<nbp; ++i)
            sortArray.push_back(std::make_pair((float)newPoints[i][axis], i));
        std::sort(sortArray.begin(), sortArray.end(), compare_pair_first<float,int>);
        helper::vector<int> old2newP;
        old2newP.resize(nbp);
        VecCoord oldPoints = newPoints.ref();
        for (int i=0; i<nbp; ++i)
        {
            newPoints[i] = oldPoints[sortArray[i].second];
            old2newP[sortArray[i].second] = i;
        }
        for (int e=0; e<nbe; ++e)
        {
            for (int i=0; i<4; i++)
                tetrahedra[e][i] = old2newP[tetrahedra[e][i]];
        }
        helper::vector< std::pair<int,int> > sortArray2;
        for (int e=0; e<nbe; ++e)
        {
            unsigned int p = tetrahedra[e][0];
            for (int i=0; i<4; i++)
                if (tetrahedra[e][i] < p) p = tetrahedra[e][i];
            sortArray2.push_back(std::make_pair(p,e));
        }
        std::sort(sortArray2.begin(), sortArray2.end(), compare_pair_first<int,int>);
        SeqTetrahedra oldTetrahedra = tetrahedra.ref();
        vector<int> oldTetraDomains = tetraDomain.ref();
        for (int i=0; i<nbe; ++i)
        {
            tetrahedra[i] = oldTetrahedra[sortArray2[i].second];
            tetraDomain[i] = oldTetraDomains[sortArray2[i].second];
        }
        break;
    }
    default: break;
    }

    helper::WriteAccessor< Data<sofa::helper::vector<Real> > > data(d_outputCellData);
    data.clear();
    if (d_label.getValue().size() != d_labelCellData.getValue().size())
    {
        helper::WriteAccessor< Data<sofa::helper::vector<Real> > > labeldata(d_labelCellData);
        labeldata.resize(d_label.getValue().size());
        msg_error(this) << "ERROR : label and labelCellData must have the same size... otherwise 0.0 will be apply for all layers";
    }
    if(d_labelCellData.getValue().size() > tetraDomain.size() + 1 )
    for (unsigned int i = 0 ; i < tetraDomain.size(); i++)
    {
        data.push_back(d_labelCellData.getValue()[tetraDomain[i]-1]);
    }

    msg_info(this) << "Generated mesh: " << nbp << " points, " << nbe << " tetrahedra.";

    d_frozen.setValue(true);

}

template <class DataTypes, class _ImageTypes>
void MeshGenerationFromImage<DataTypes, _ImageTypes>::draw(const sofa::core::visual::VisualParams* vparams)
{
    if (d_drawTetras.getValue())
    {
        helper::ReadAccessor< Data<VecCoord> > x = d_newX0;
        helper::ReadAccessor< Data<SeqTetrahedra> > tetrahedra = d_tetrahedra;
        helper::ReadAccessor< Data< vector<int> > > tetraDomain = d_tetraDomain;

        vparams->drawTool()->setLightingEnabled(false);
        std::vector< std::vector<defaulttype::Vector3> > pointsDomains[4];
        for(unsigned int i=0; i<4; ++i)
            pointsDomains[i].resize(m_tetraDomainLabels.size());
        int domainLabel = 0;
        for(unsigned int i=0; i<tetrahedra.size(); ++i)
        {            
            domainLabel =  std::find(m_tetraDomainLabels.begin(), m_tetraDomainLabels.end(),tetraDomain[i]) - m_tetraDomainLabels.begin();

            int a = tetrahedra[i][0];
            int b = tetrahedra[i][1];
            int c = tetrahedra[i][2];
            int d = tetrahedra[i][3];
            Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
            Coord pa = (x[a]+center)*(Real)0.666667;
            Coord pb = (x[b]+center)*(Real)0.666667;
            Coord pc = (x[c]+center)*(Real)0.666667;
            Coord pd = (x[d]+center)*(Real)0.666667;

            pointsDomains[0][domainLabel].push_back(pa);
            pointsDomains[0][domainLabel].push_back(pb);
            pointsDomains[0][domainLabel].push_back(pc);

            pointsDomains[1][domainLabel].push_back(pb);
            pointsDomains[1][domainLabel].push_back(pc);
            pointsDomains[1][domainLabel].push_back(pd);

            pointsDomains[2][domainLabel].push_back(pc);
            pointsDomains[2][domainLabel].push_back(pd);
            pointsDomains[2][domainLabel].push_back(pa);

            pointsDomains[3][domainLabel].push_back(pd);
            pointsDomains[3][domainLabel].push_back(pa);
            pointsDomains[3][domainLabel].push_back(pb);
        }

        for(size_t i=0; i<m_tetraDomainLabels.size(); i++)
        {
            vparams->drawTool()->drawTriangles(pointsDomains[0][i], defaulttype::Vec<4,float>(fmod(i*0.5,1.5), fmod(0.5-fmod(i*0.5,1.5),1.5)-0.1, 1.0-fmod(i*0.5,1.5), 1));
            vparams->drawTool()->drawTriangles(pointsDomains[1][i], defaulttype::Vec<4,float>(fmod(i*0.5,1.5)-0.1, fmod(0.5-fmod(i*0.5,1.5),1.5)-0.2, 1.0-fmod(i*0.5,1.5), 1));
            vparams->drawTool()->drawTriangles(pointsDomains[2][i], defaulttype::Vec<4,float>(fmod(i*0.5,1.5)-0.2, fmod(0.5-fmod(i*0.5,1.5),1.5)-0.3, 1.0-fmod(i*0.5,1.5), 1));
            vparams->drawTool()->drawTriangles(pointsDomains[3][i], defaulttype::Vec<4,float>(fmod(i*0.5,1.5)-0.3, fmod(0.5-fmod(i*0.5,1.5),1.5), 1.0-fmod(i*0.5,1.5), 1));
        }
    }
}

} //cgal

#endif //CGALPLUGIN_MESHGENERATIONFROMIMAGE_INL
