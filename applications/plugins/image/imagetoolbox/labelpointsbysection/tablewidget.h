#ifndef TABLEWIDGET_H
#define TABLEWIDGET_H

#include <QGroupBox>
#undef None // conflict with X11/X.h
#include <QTableWidget>
#include <QPushButton>
#include <QComboBox>
#include <QMap>
#include <QVBoxLayout>

#include <sofa/defaulttype/VecTypes.h>
#include <iostream>



class TableWidgetForLabelPointBySectionToolBoxAction: public QGroupBox
{
Q_OBJECT

public:
    typedef sofa::defaulttype::Vec3f Vec3f;

    struct Point
    {
        Vec3f ip;
        Vec3f p;
    };

    typedef sofa::helper::vector<Point> VecCoord;
    typedef QMap<unsigned int, VecCoord > MapSection;


private:

    QComboBox * listSection;
    QTableWidget * listPoints;
    QPushButton * deteteSection;
    QPushButton * deletePoint;
    QPushButton * moveupPoint;
    QPushButton * movedownPoint;
    
    MapSection * mapSection;
    unsigned int currentSection;
    
public:
    
    TableWidgetForLabelPointBySectionToolBoxAction(QWidget *parent=NULL): QGroupBox(parent)
    {
        listSection = new QComboBox();
        listPoints = new QTableWidget();
        deteteSection  = new QPushButton("del section");
        deletePoint = new QPushButton("del");
        deletePoint->setEnabled(false);
        moveupPoint = new QPushButton("up");
        moveupPoint->setEnabled(false);
        movedownPoint = new QPushButton("down");
        movedownPoint->setEnabled(false);
    
        QVBoxLayout *vlayout = new QVBoxLayout();
    
        QHBoxLayout *hlayout = new QHBoxLayout();
        hlayout->addWidget(listSection);
        hlayout->addWidget(deteteSection);
        this->connect(deteteSection,SIGNAL(clicked()),this,SLOT(deleteSectionAction()));
    
        vlayout->addLayout(hlayout);
        vlayout->addWidget(listPoints);
    
        QHBoxLayout *hlayout2 = new QHBoxLayout();
        hlayout2->addWidget(deletePoint);
        hlayout2->addWidget(moveupPoint);
        hlayout2->addWidget(movedownPoint);
    
        vlayout->addLayout(hlayout2);
        
        this->setTitle("Sections");
        this->setLayout(vlayout);
        
        listPoints->insertColumn(0);
        listPoints->setHorizontalHeaderItem (0, new QTableWidgetItem("X") );
        listPoints->insertColumn(1);
        listPoints->setHorizontalHeaderItem (1, new QTableWidgetItem("Y") );
        listPoints->insertColumn(2);
        listPoints->setHorizontalHeaderItem (2, new QTableWidgetItem("Z") );
        listPoints->setSelectionBehavior(QAbstractItemView::SelectRows);
        listPoints->setSelectionMode(QAbstractItemView::SingleSelection);
        listPoints->setEditTriggers(QAbstractItemView::NoEditTriggers);
        
        connect(listSection,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxchangeSection(int)));

        
        //updateData();
    }

    inline void setMapSection(MapSection *m){mapSection=m;updateData();}
    
    void updateData()
    {
        if(!mapSection)
        {
            return;
        }
        disconnect(listSection,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxchangeSection(int)));

        // list the available slides
        listSection->clear();

        QList<unsigned int> key = mapSection->keys();


        int j=0;
        for(int i=0;i<key.size();i++)
        {

            if(currentSection!=key[i] && (mapSection->value(key[i])).size()==0)
            {
                mapSection->remove(key[i]);

            }
            else
            {

                listSection->addItem(QString::number(key[i]));
                if(currentSection==key[i])
                {

                    listSection->setCurrentIndex(j);
                }
                j++;
            }
        }
        
        // list points of the current section
        VecCoord &vector = mapSection->operator [](currentSection);
        
        while(listPoints->rowCount()!=0)
            listPoints->removeRow(0);
        

        for(unsigned int i=0;i<vector.size();i++)
        {
            listPoints->insertRow(i);
            listPoints->setItem(i,0,new QTableWidgetItem(QString::number(vector[i].ip.x())));
            listPoints->setItem(i,1,new QTableWidgetItem(QString::number(vector[i].ip.y())));
            listPoints->setItem(i,2,new QTableWidgetItem(QString::number(vector[i].ip.z())));
            
        }

        listPoints->resizeColumnsToContents();
        connect(listSection,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxchangeSection(int)));
    }

    void setSection(unsigned int cs)
    {
        if(currentSection!=cs)
        {

                currentSection = cs;
                updateData();
        }


    }

private slots:
    void comboBoxchangeSection(int i)
    {
        currentSection = mapSection->keys()[i];
        updateData();
        emit changeSection(currentSection);
    }

    void deleteSectionAction()
    {
        VecCoord &v = mapSection->operator [](currentSection);
        v.clear();
        updateData();
    }
    
signals:
    void changeSection(int);
    void update();
    void reload();
    void saveFile();
    void loadFile();
    
};


#endif // TABLEWIDGET_H
