import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Controls.Private 1.0
import QtQuick.Controls.Styles 1.1
import QtQuick.Layouts 1.0

Item {
    id: groupbox

    /*! \internal */
    default property alias __content: container.data

    readonly property alias contentItem: container

    implicitWidth: (!anchors.fill ? container.calcWidth() : 0)
    implicitHeight: (!anchors.fill ? container.calcHeight() : 0)

    Layout.minimumWidth: implicitWidth
    Layout.minimumHeight: implicitHeight

    Accessible.role: Accessible.Grouping
    Accessible.name: title

    activeFocusOnTab: false

    data: [
        Item {
            id: container
            objectName: "container"
            z: 1
            focus: true
            anchors.fill: parent

            property Item layoutItem: container.children.length === 1 ? container.children[0] : null
            function calcWidth () { return (layoutItem ? (layoutItem.implicitWidth) +
                                                         (layoutItem.anchors.fill ? layoutItem.anchors.leftMargin +
                                                                                    layoutItem.anchors.rightMargin : 0) : container.childrenRect.width) }
            function calcHeight () { return (layoutItem ? (layoutItem.implicitHeight) +
                                                          (layoutItem.anchors.fill ? layoutItem.anchors.topMargin +
                                                                                     layoutItem.anchors.bottomMargin : 0) : container.childrenRect.height) }
        }]
}
