#include <GL/glut.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "traqueboule.h"

#include <sofa/helper/ArgumentParser.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/component/typedef/Sofa_typedef.h>
#include <sofa/gui/SofaGUI.h>

using sofa::core::componentmodel::behavior::OdeSolver;
using sofa::component::odesolver::EulerSolver;
using sofa::component::odesolver::CGImplicitSolver;
using sofa::component::topology::MeshTopology;
using sofa::component::visualmodel::OglModel;


#include "../lib/BoostSceneGraph.h"
#include "../lib/BoostSystem.h"

namespace
{
}
// lumière 0
GLfloat l0_position[] = {0,0,0,1};
GLfloat l0_ambient[] = {0.1,0.1,0.1};
GLfloat l0_diffuse[] = {1,1,1};
GLfloat l0_specular[] = {1,1,1};

// lumière 1
GLfloat l1_position[] = {0,0,0,1};
GLfloat l1_ambient[] = {0.1,0.1,0.1};
GLfloat l1_diffuse[] = {1,1,1};
GLfloat l1_specular[] = {1,1,1};



BoostSceneGraph sceneGraph;
typedef BoostSceneGraph::Node Node;

void buildScene( BoostSceneGraph& sceneGraph );
void buildScene( BoostSceneGraph& sceneGraph )
{
    Node node1 = sceneGraph.addNode();
    sceneGraph.systemMap[node1]->setName("n1");

    // Solver
    //EulerSolver* solver = new EulerSolver;
    OdeSolver* solver = new CGImplicitSolver;
    sceneGraph.systemMap[node1]->addObject(solver);
    //solver->f_printLog.setValue(true);

    // Tetrahedron degrees of freedom
    MechanicalObject3* DOF = new MechanicalObject3;
    sceneGraph.systemMap[node1]->addObject(DOF);
    DOF->resize(4);
    DOF->setName("DOF");
    VecCoord3& x = *DOF->getX();

    x[0] = Coord3(0,10,0);
    x[1] = Coord3(10,0,0);
    x[2] = Coord3(-10*0.5,0,10*0.866);
    x[3] = Coord3(-10*0.5,0,-10*0.866);

    // Tetrahedron uniform mass
    UniformMass3* mass = new UniformMass3;
    //BoostSceneGraph::MassMap  gmass = sceneGraph.getMass();
    sceneGraph.systemMap[node1]->addObject(mass);
    mass->setMass(2);
    mass->setName("mass");

    // Tetrahedron topology
    MeshTopology* topology = new MeshTopology;
    sceneGraph.systemMap[node1]->addObject(topology);
    topology->setName("topology");
    topology->addTetrahedron(0,1,2,3);

    // Tetrahedron constraints
    FixedConstraint3* constraints = new FixedConstraint3;
    sceneGraph.systemMap[node1]->addObject(constraints);
    constraints->setName("constraints");
    constraints->addConstraint(0);

    // Tetrahedron force field
    TetrahedronFEMForceField3* spring = new  TetrahedronFEMForceField3;
    sceneGraph.systemMap[node1]->addObject(spring);
    spring->setUpdateStiffnessMatrix(true);
    spring->setYoungModulus(20);
    spring->f_updateStiffnessMatrix.setValue(true);

    // Tetrahedron skin
    Node n2 = sceneGraph.addNode("n2");
    sceneGraph.addEdge(node1,n2);
    BoostSystem* skin = sceneGraph.systemMap[n2];

    // The visual model
    OglModel* visual = new OglModel();
    visual->setName( "visual" );
    visual->load(sofa::helper::system::DataRepository.getFile("VisualModels/liver-smooth.obj"), "", "");
    visual->setColor("red");
    visual->applyScale(0.7);
    visual->applyTranslation(1.2, 0.8, 0);
    skin->addObject(visual);

    // The mapping between the tetrahedron (DOF) and the liver (visual)
    BarycentricMapping3_to_Ext3* mapping = new BarycentricMapping3_to_Ext3(DOF, visual);
    mapping->setName( "mapping" );
    skin->addObject(mapping);

    sceneGraph.init();
    sceneGraph.setShowBehaviorModels(true);
    sceneGraph.setShowVisualModels(true);
    //sceneGraph.setShowNormals(true);

}


// Actions d'affichage
void display(void);
void display(void)
{
    // Details sur le mode de tracé
    glEnable( GL_DEPTH_TEST );            // effectuer le test de profondeur
    glDisable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT,GL_FILL);
    //glPolygonMode(GL_BACK,GL_LINE);
    //glShadeModel(GL_FLAT);
    glEnable(GL_NORMALIZE);

    // Effacer tout
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT); // la couleur et le z

    glLoadIdentity();  // repere camera

    glLightfv( GL_LIGHT0, GL_POSITION,  l0_position );
    glLightfv( GL_LIGHT1, GL_POSITION,  l1_position );

    tbVisuTransform(); // origine et orientation de la scene

    sceneGraph.glDraw();

    glColor3f(1,1,1);
    glutWireCube( 10 );


    glutSwapBuffers();
}

// pour changement de taille ou desiconification
void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glOrtho (-1.1, 1.1, -1.1,1.1, -1000.0, 1000.0);
    gluPerspective (50, (float)w/h, 1, 100);
    glMatrixMode(GL_MODELVIEW);
}

// prise en compte du clavier
void keyboard(unsigned char key, int x, int y)
{
    printf("key %d pressed at %d,%d\n",key,x,y);
    fflush(stdout);
    switch (key)
    {
    case 27:     // touche ESC
        exit(0);
    }
}

void animate();
void animate()
{
    sceneGraph.animate(0.04);
    glutPostRedisplay();
}


// programme principal
int main(int argc, char** argv)
{

    sofa::gui::SofaGUI::Init(argv[0]);

    int W_fen = 600;  // largeur fenetre
    int H_fen = 600;  // hauteur fenetre

    sofa::helper::parse("Comparaison de maillages. Voici la liste des options: ")
    .option(&W_fen,'L',"Largeur","largeur de la fenêtre en pixels")
    .option(&H_fen,'H',"Hauteur","hauteur de la fenêtre en pixels")
    (argc,argv);

    glutInit(&argc, argv);

    buildScene( sceneGraph );

    // couches du framebuffer utilisees par l'application
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );

    // position et taille de la fenetre
    glutInitWindowPosition(200, 100);
    glutInitWindowSize(W_fen,H_fen);
    glutCreateWindow(argv[0]);


    // Initialisation du point de vue
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0,0,-20);
    tbInitTransform();     // initialisation du point de vue
    tbHelp();                      // affiche l'aide sur la traqueboule

    // cablage des callback
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutMouseFunc(tbMouseFunc);    // traqueboule utilise la souris
    glutMotionFunc(tbMotionFunc);  // traqueboule utilise la souris
    glutIdleFunc( animate );

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);

    // lumiere 0
    glLightfv( GL_LIGHT0, GL_AMBIENT,   l0_ambient );
    glLightfv( GL_LIGHT0, GL_DIFFUSE,   l0_diffuse );
    glLightfv( GL_LIGHT0, GL_SPECULAR,  l0_specular );

    // lumiere 1
    glLightfv( GL_LIGHT1, GL_AMBIENT,   l1_ambient );
    glLightfv( GL_LIGHT1, GL_DIFFUSE,   l1_diffuse );
    glLightfv( GL_LIGHT1, GL_SPECULAR,  l1_specular );

    // lancement de la boucle principale
    glutMainLoop();
    return 0;  // instruction jamais exécutée
}

