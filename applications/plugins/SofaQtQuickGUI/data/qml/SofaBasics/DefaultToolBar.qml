import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import "qrc:/SofaCommon/SofaCommonScript.js" as SofaCommonScript
import "qrc:/SofaCommon/SofaApplicationScript.js" as SofaApplicationScript

ToolBar {
    id: root
    implicitHeight: 45

    property Scene scene

    TabView {
        Tab {
            title: "Interaction"

            Row {
                id: interactorPositioner

                Component {
                    id: interactorButtonComponent

                    ToolButton {
                        property string interactorName
                        property Component interactorComponent

                        text: interactorName
                        checkable: true
                        checked: SofaApplicationScript.Application.interactorComponent ? interactorName === SofaApplicationScript.Application.interactorName : false
                        onCheckedChanged: SofaApplicationScript.Application.interactorComponent = interactorComponent
                        onClicked: checked = true;
                    }
                }

                Connections {
                    target: SofaApplicationScript.Application
                    onInteractorComponentMapChanged: interactorPositioner.update();
                }

                function update() {
                    for(var i = 0; i < children.length; ++i)
                        children[i].destroy();

                    var interactorComponentMap = SofaApplicationScript.Application.interactorComponentMap;
                    for(var key in interactorComponentMap)
                        if(interactorComponentMap.hasOwnProperty(key))
                            SofaCommonScript.InstanciateComponent(interactorButtonComponent, interactorPositioner, {interactorName: key, interactorComponent: interactorComponentMap[key]});
                }
            }
        }

        Tab {
            title: "Misc"

            Row {

                ToolButton {
                    text: "A"
                }

                ToolButton {
                    text: "B"
                }

                ToolButton {
                    text: "C"
                }
            }
        }

        style: TabViewStyle {
            frameOverlap: 0
            tabOverlap: -5

            tab: Rectangle {
                implicitWidth: Math.max(text.implicitWidth + 4, 80)
                implicitHeight: 20

                radius: 5
                gradient: Gradient {
                    GradientStop {color: "#EEE"; position: 0.0}
                    GradientStop {color: "#DDD"; position: 0.5}
                    GradientStop {color: styleData.selected ? "#BBB" : "#CCC" ; position: 1.0}
                }

                Text {
                    id: text
                    anchors.centerIn: parent
                    text: styleData.title
                    color: styleData.selected ? "black" : "grey"
                    font.bold: styleData.selected
                }
            }

            frame: null
        }
    }
}
