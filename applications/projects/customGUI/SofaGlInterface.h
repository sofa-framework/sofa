#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/core/visual/DrawToolGL.h>

#include "PickedPoint.h"
#include "SpringInteractor.h"


using namespace sofa;
using simulation::Node;
typedef sofa::simulation::graph::DAGSimulation ParentSimulation;
//typedef sofa::simulation::tree::TreeSimulation ParentSimulation;

namespace sofa {
namespace newgui {


// sofa types should not be exposed
typedef sofa::defaulttype::Vector3 Vec3;
typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;





/** Prototype used to completely encapsulate the use of Sofa in an OpenGL application, without any standard Sofa GUI.
 *
 * @author Francois Faure, 2014
 * */
class SofaGlInterface : public ParentSimulation
{
    // sofa types should not be exposed
    typedef sofa::defaulttype::Vector3 Vec3;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;

    Node::SPtr groot; ///< root of the graph
    Node::SPtr sroot; ///< root of the scene, child of groot


    GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];



public:

    typedef sofa::newgui::Interactor Anchor;

    bool debug;

    SofaGlInterface();

    /**
     * @brief Initialize Sofa and load a scene file
     * @param plugins List of plugins to load
     * @param fileName Scene file to load
     */
    void init( std::vector<std::string>& plugins, const std::string& fileName );

    void printScene();

    void reshape(int,int);

    /**
     * @brief glDraw Draw the Sofa scene using OpenGL.
     * Requires that an OpenGL context is active.
     */
    void glDraw();

    /**
     * @brief Integrate time by one step and update the Sofa scene.
     */
    void animate();

    Node::SPtr getRoot();

    PickedPoint glRayPick( GLdouble ox, GLdouble oy, GLdouble oz, int x, int y );

    void attach( Anchor* anchor );

    void move( Anchor* anchor, int x, int y);

    void detach( Anchor* anchor);


protected:
    sofa::core::visual::DrawToolGL   drawToolGL;
    sofa::core::visual::VisualParams* vparams;


};



}// newgui
}// sofa

