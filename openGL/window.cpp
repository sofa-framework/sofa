#include "window.h"
#include <QImage>

Window::Window(QGLWidget *parent) : QGLWidget(parent)
{
}
// material
GLfloat m_emission[] =  {0, 0, 0};
GLfloat m_ambient[] =   {1,1,1};
GLfloat m_diffuse[] =   {0.5,0.5,0.5};
GLfloat m_specular[] =  {0.5,0.5,0.5};
GLfloat m_shininess =   20.0;

// global lighting parameters
GLfloat l_ambient[] = {0.1,0.1,0.1};
GLint l_localViewer = GL_TRUE;
GLint l_twoSides = GL_TRUE;
//GLint l_separateDiffuseAndSpecular = GL_SINGLE_COLOR;

// light source 0
bool lumiere0 = true;
GLfloat l0_position[] = {0,0,0,1};
GLfloat l0_ambient[] = {0.1,0.1,0.1};
GLfloat l0_diffuse[] = {1,0,0};
GLfloat l0_specular[] = {1,0,0};

// light source 1
bool lumiere1 = true;
GLfloat l1_position[] = {0,0,0,1};
GLfloat l1_ambient[] = {0.1,0.1,0.1};
GLfloat l1_diffuse[] = {1,0,0};
GLfloat l1_specular[] = {0,1,0};
GLfloat lattitude1 = 0;
GLfloat longitude1 = 0;
GLfloat distance1 = 2;

void Window::affichage()
{
                GLfloat position[] = {2, 1, 3, 1};
//                ---	Initialisation	---
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
                glPushMatrix();
//			glRotated( . . . );
//			glTranslated( . . . );
                        glLightfv(GL_LIGHT0, GL_POSITION, position);
                glPopMatrix();
                gluLookAt(0,0,0,0,0,0,1,1,1);		// Définit la matrice de visualisation et la
                                        // multiplie à droite de la matrice active.
//                --- Tracé de l’objet.
//                --- Redessine l’objet fixe avec la lumière modifiée. 	---

                void glutSwapBuffers();
}


void Window::initializeGL()
{
    f_x = 0.0;
    loadTexture("texture/box.png");

    glEnable(GL_TEXTURE_2D);
    glShadeModel(GL_SMOOTH);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glDisable(GL_LIGHTING);

    glDepthFunc(GL_LEQUAL);

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

}
void Window::resizeGL(int width, int height)
{
    if(height == 0)
        height = 1;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)width/(GLfloat)height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void Window::paintGL()
{
    f_x += 0.1;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(-1.5f, 0.0f, -6.0f);
    glRotatef(f_x, 1.0, 0.3, 0.1);

    glBindTexture(GL_TEXTURE_2D, texture[0]);

   glBegin(GL_QUADS);//dessine cube
       // Face Avant
       glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
       // Face Arrière
       glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
       // Face Haut
       glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
       // Face Bas
       glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
       // Face Droite
       glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
       // Face Gauche
       glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
   glEnd();
//   glBegin(GL_LINE_LOOP);//pyramide
//
//    glVertex3f(0.5,-0.5,0.0);
//    glVertex3f(0.5,0.5,0.0);
//    glVertex3f(-0.5,0.5,0.0);
//    glVertex3f(-0.5,-0.5,0.0);
//    glEnd();
//    //draw the nose
//    glBegin(GL_LINES);
//
//    glVertex3f(0.5,-0.5,0.0);
//    //glColor3f(1.0,0.0,0.0);
//    glVertex3f(0.0,0.0,1);
//
//    //glColor3f(1.0,1.0,1.0);
//    glVertex3f(0.5,0.5,0.0);
//    glColor3f(1.0,0.0,0.0);
//    glVertex3f(0.0,0.0,1);
//
//    glColor3f(1.0,1.0,1.0);
//    glVertex3f(-0.5,0.5,0.0);
//    glColor3f(1.0,0.0,0.0);
//    glVertex3f(0.0,0.0,1);
//
//    glColor3f(1.0,1.0,1.0);
//    glVertex3f(-0.5,-0.5,0.0);
//    glColor3f(1.0,0.0,0.0);
//    glVertex3f(0.0,0.0,1);
//    glEnd();

}

void Window::loadTexture(QString textureName)
{
    QImage qim_Texture;
    QImage qim_TempTexture;
    textureName = "C:/Users/Public/Pictures/boi-mur.jpg";
    qim_TempTexture.load(textureName);
    qim_Texture = QGLWidget::convertToGLFormat( qim_TempTexture );
    glGenTextures( 1, &texture[0] );
    glBindTexture( GL_TEXTURE_2D, texture[0] );
    glTexImage2D( GL_TEXTURE_2D, 0, 3, qim_Texture.width(), qim_Texture.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, qim_Texture.bits() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
}
void Window::InitGL()
{
        glEnable(GL_DEPTH_TEST); 	// Active le test de profondeur
        glEnable(GL_LIGHTING); 	// Active l'éclairage
        glEnable(GL_LIGHT0); 	// Allume la lumière n°1
        glDisable(GL_LIGHTING);
}

Window::~Window()
{

}
