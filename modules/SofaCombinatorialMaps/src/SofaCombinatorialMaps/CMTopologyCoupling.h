#ifndef SOFA_COMPONENT_TOPOLOGY_CMTOPOLOGYCOUPLING_H
#define SOFA_COMPONENT_TOPOLOGY_CMTOPOLOGYCOUPLING_H


#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/gl.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <SofaCombinatorialMaps/SurfaceTopologyContainer.h>
#include <SofaBaseTopology/MeshTopology.h>

#include <SofaCombinatorialMaps/SurfaceMaskTraversal.h>

namespace sofa
{

namespace component
{

namespace topology
{

template <class DataTypes>
class CMTopologyCoupling : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CMTopologyCoupling, DataTypes), BaseObject);

    using uint32 = std::uint32_t;
    using Real = typename DataTypes::Real;
    using Coord = typename DataTypes::Coord;
    using VecCoord = typename DataTypes::VecCoord;
    using Index = uint32;

    using SurfaceTopology = sofa::component::topology::SurfaceTopologyContainer;
    using FilteredQuickTraversor = SurfaceTopology::FilteredQuickTraversor;
    using Face = SurfaceTopology::Face;
    template<typename T>
    using FaceAttribute = typename SurfaceTopology::Topology::template FaceAttribute<T>;

    using Arc = MeshTopology::Edge;
    using SeqArcs = MeshTopology::SeqEdges;

protected:
    CMTopologyCoupling()
        :BaseObject()
        , surface_positions_(initData(&surface_positions_,"surfacePositions","Surface topology positions"))
        , surface_(initLink("surface", "Surface topology"))
        , graph_(initLink("graph", "Graph topology (MeshTopology type)"))
//        , mask_(initLink("mask", "Traversal mask"))
    {}

    ///\param[out] q the closest point to p on the line segment from A to B
    ///\retval true if q is inbetween the segment AB, false otherwise
    bool project_point_to_line_segment(Coord& A,Coord& B,Coord& p, Coord& q)
    {
        bool is_inbetween=false;
        //vector from A to B
        Coord AB = (B-A);

        //squared distance from A to B
        Real AB_squared = dot(AB, AB);

        if(AB_squared == 0)
        {
            //A and B are the same point
            q = A;
            is_inbetween = true;
        }
        else
        {
            //vector from A to p
            Coord Ap = (p-A);
            //from http://stackoverflow.com/questions/849211/
            //Consider the line extending the segment, parameterized as A + t (B - A)
            //We find projection of point p onto the line.
            //It falls where t = [(p-A) . (B-A)] / |B-A|^2
            Real t = dot(Ap, AB)/AB_squared;
            if (t < 0.0)
                //"Before" A on the line, just return A
                q = A;
            else if (t > 1.0)
                //"After" B on the line, just return B
                q = B;
            else
            {
                //projection lines "inbetween" A and B on the line
                q = A + t * AB;
                is_inbetween = true;
            }
        }

        return is_inbetween;
    }

    /// \details not very robust to geometric protusions and surface combinatorics (triangle only)
    /// \todo improve the coupling algorithm
    void computeCoupling()
    {
        helper::ReadAccessor< Data<VecCoord> > pos = surface_positions_;
        helper::WriteAccessor< Data<FaceAttribute<uint32>>> tag_ = face_tag_;

        const SeqArcs arcs = graph_->getEdges();
        surface_->foreach_cell([&](Face f)
        {
            //compute m_f the center of f = 1/card(f) * (sum (f_i))
            const auto& dofs = surface_->get_dofs(f);

            Coord centroid = 1./3. * (pos[dofs[0]] +pos[dofs[1]] + pos[dofs[2]]);

            Real max = std::numeric_limits<Real>::max();
            for(unsigned int i = 0 ; i < arcs.size() ; ++i)
            {
                Coord A(graph_->getPosX(arcs[i][0]), graph_->getPosY(arcs[i][0]), graph_->getPosZ(arcs[i][0]));
                Coord B(graph_->getPosX(arcs[i][1]), graph_->getPosY(arcs[i][1]), graph_->getPosZ(arcs[i][1]));
                Coord q;
                project_point_to_line_segment(A, B, centroid, q);

                Real l = (centroid-q).norm();
                if(l < max)
                {
                    max = l;
                    tag_[f.dart] = i;
                }
            }
        });
    }

    void jet(const float f, float& r, float& g, float& b)
    {
        // Only important if the number of colors is small.
        // In which case the rest is still wrong anyway
        // x = linspace(0,1,jj)' * (1-1/jj) + 1/jj;
        const float rone = 0.8;
        const float gone = 1.0;
        const float bone = 1.0;
        float x = f;
        x = (f<0 ? 0 : (x>1 ? 1 : x));

        if (x<1. / 8.)
        {
            r = 0;
            g = 0;
            b = bone*(0.5 + (x) / (1. / 8.)*0.5);
        }
        else if (x<3. / 8.)
        {
            r = 0;
            g = gone*(x - 1. / 8.) / (3. / 8. - 1. / 8.);
            b = bone;
        }
        else if (x<5. / 8.)
        {
            r = rone*(x - 3. / 8.) / (5. / 8. - 3. / 8.);
            g = gone;
            b = (bone - (x - 3. / 8.) / (5. / 8. - 3. / 8.));
        }
        else if (x<7. / 8.)
        {
            r = rone;
            g = (gone - (x - 5. / 8.) / (7. / 8. - 5. / 8.));
            b = 0;
        }
        else
        {
            r = (rone - (x - 7. / 8.) / (1. - 7. / 8.)*0.5);
            g = 0;
            b = 0;
        }
    }

public:
    virtual void init() override
    {
//        cell_traversor = cgogn::make_unique<FilteredQuickTraversor>(surface_->getMap());

//        if (!mask_.get())
//            cell_traversor->build<Face>();
//        else
//        {
//            cell_traversor->build<Face>();
//            cell_traversor->set_filter<Face>(std::move(std::ref(*(mask_.get()))));
//        }

        face_tag_ = surface_->template add_attribute<uint32, Face>("CMTopologyCoupling_face_tag");
        computeCoupling();
    }

    void draw(const core::visual::VisualParams* vparams) override
    {
        using std::to_string;

        if (!vparams->displayFlags().getShowVisualModels()) return;

        helper::vector<defaulttype::Vector3> points;
        helper::vector<defaulttype::Vector3> normals;
        helper::vector<defaulttype::Vec4f> colors;

        uint32 nb_edges = graph_->getNbEdges();
        const SeqArcs arcs = graph_->getEdges();
        helper::ReadAccessor< Data<VecCoord> > pos = surface_positions_;

        helper::WriteAccessor< Data<FaceAttribute<uint32>>> tag_ = face_tag_;


        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0,true);

        surface_->foreach_cell([&](Face face)
        {
            const auto& dofs = surface_->get_dofs(face);

            Index x = dofs[0];
            Index y = dofs[1];
            Index z = dofs[2];

            points.push_back(defaulttype::Vector3(pos[x][0], pos[x][1], pos[x][2]));
            points.push_back(pos[y]);
            points.push_back(pos[z]);

            defaulttype::Vector3 N = (pos[y] - pos[x]).cross((pos[z] - pos[x]));
            normals.push_back(N);


            float factor = float(tag_[face.dart]) / float(nb_edges);
            float r, g, b;
            jet(factor, r, g, b);
            //std::cout << "factor: " << factor << " | (" << r << "," << g << "," << b << ")" << std::endl;
            colors.push_back(defaulttype::Vec4f(r,g,b,1.));

            Coord centroid = 1./3. * (pos[dofs[0]] +pos[dofs[1]] + pos[dofs[2]]);
            Arc arc = arcs[tag_[face.dart]];

            defaulttype::Vector3 A(graph_->getPosX(arc[0]), graph_->getPosY(arc[0]), graph_->getPosZ(arc[0]));
            defaulttype::Vector3 B(graph_->getPosX(arc[1]), graph_->getPosY(arc[1]), graph_->getPosZ(arc[1]));

            vparams->drawTool()->drawLine(centroid, (A+B)/2., defaulttype::Vec4f(1.,0.,0.,1.) );
            vparams->drawTool()->draw3DText(centroid, 1.0, defaulttype::Vec4f(1.,0.,0.,1.), to_string(tag_[face.dart]).c_str());
        });

//       vparams->drawTool()->drawTriangles(points, normals, colors);



        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0,false);

    }

public:
    Data<FaceAttribute<uint32>> face_tag_;
    Data<VecCoord> surface_positions_;
//    std::unique_ptr<FilteredQuickTraversor> cell_traversor;

private:
    SingleLink<CMTopologyCoupling<DataTypes>, SurfaceTopology, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> surface_;
    SingleLink<CMTopologyCoupling<DataTypes>, MeshTopology, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> graph_;
//    SingleLink<CMTopologyCoupling<DataTypes>, sofa::component::topology::SurfaceMaskTraversal, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mask_;

};

} //end namespace topology

} //end namespace component

} //end namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_SURFACEMASKTRAVERSAL_H
