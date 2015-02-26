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
    property var sceneComponent: scene && scene.ready ? scene.listModel.getComponentById(scene.listModel.selectedId) : null

    enabled: scene ? scene.ready : false

    GridLayout {
        id: layout
        anchors.fill: parent
        columns: 1

        /*Text {
            text: "Index: " + scene && -1 !== scene.listModel.selectedItem ? scene.listModel.selectedItem : ""
        }

        Text {
            text: "Name: " + scene && -1 !== scene.listModel.selectedItem ? scene.listModel.get(scene.listModel.selectedItem)["name"] : ""
        }*/

        Component {
            id: delegate

            Data {
                anchors.left: parent.left
                anchors.right: parent.right
                //height: 16

                scene: root.scene
                sceneData: dataListModel.getDataById(index)

                nameLabelWidth: scrollView.nameLabelImplicitWidth
                Component.onCompleted: updateNameLabelWidth();
                onNameLabelImplicitWidthChanged: updateNameLabelWidth();

                function updateNameLabelWidth() {
                    scrollView.nameLabelImplicitWidth = Math.max(scrollView.nameLabelImplicitWidth, nameLabelImplicitWidth);
                }
            }
        }

        ScrollView {
            id: scrollView
            Layout.fillWidth: true
            Layout.preferredHeight: 400
            clip: true

            property int nameLabelImplicitWidth : 16

            ListView {
                id: listView
                width: parent.width
                model: DataListModel {
                    id: dataListModel
                    sceneComponent: root.sceneComponent
                }

                delegate: delegate
                focus: true
            }
        }
    }
}
