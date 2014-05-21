import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0

Rectangle {
    id: base

    color: Qt.rgba(1, 1, 0, 0.7)

    ScrollView {
        anchors.fill: parent

        ListView {
            model: VisualDataModel {
                id: visualModel
                model: ListModel {
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                    ListElement { name: "MechanicalObject" }
                    ListElement { name: "UniformMass" }
                    ListElement { name: "Topology" }
                }
                delegate: Text {
                    text: "> " + name
                }
            }
        }
    }
}
