import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2
import SofaBasics 1.0
import SceneComponent 1.0
import DataListModel 1.0

CollapsibleGroupBox {
    id: root
    implicitWidth: 0

    title: "Data"
    property int priority: 80

    property Scene scene
    property var sceneComponent: scene && scene.ready ? scene.listModel.getComponentById(scene.listModel.selectedId) : null // TODO: use SceneGraphView.selectedId

    enabled: scene ? scene.ready : false

    GridLayout {
        id: layout
        anchors.fill: parent
        columns: 1

        ListView {
            id: listView
            Layout.fillWidth: true
            Layout.preferredHeight: 400
            clip: true
            focus: true

            property int nameLabelImplicitWidth : 16

            model: DataListModel {
                id: dataListModel
                sceneComponent: root.sceneComponent

                onSceneComponentChanged: listView.nameLabelImplicitWidth = 16;
            }

            section.property: "group"
            section.criteria: ViewSection.FullString
            section.delegate: Rectangle {
                width: parent.width
                height: childrenRect.height
                color: "darkgrey"

                Text {
                    width: parent.width
                    horizontalAlignment: Text.AlignHCenter
                    text: section
                    font.bold: true
                    font.pixelSize: 16
                }
            }

            delegate: Item {
                anchors.left: parent.left
                anchors.right: parent.right
                height: data.height

                Rectangle {
                    visible: data.modified
                    anchors.fill: data
                    color: "lightsteelblue"
                }

                Data {
                    id: data
                    anchors.left: parent.left
                    anchors.right: parent.right

                    scene: root.scene
                    sceneData: dataListModel.getDataById(index)

                    nameLabelWidth: listView.nameLabelImplicitWidth
                    Component.onCompleted: updateNameLabelWidth();
                    onNameLabelImplicitWidthChanged: updateNameLabelWidth();

                    //onModifiedChanged: ; // TODO: update SceneGraphView (component name may have changed)

                    function updateNameLabelWidth() {
                        listView.nameLabelImplicitWidth = Math.max(listView.nameLabelImplicitWidth, nameLabelImplicitWidth);
                    }
                }
            }
        }
    }
}
