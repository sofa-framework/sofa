import Qt.labs.settings 1.0

Settings {
    category: "recent"

    property string scenes      // recently opened scenes

    function add(path) {
        scenes = path + ";" + scenes.replace(path + ";", "");
    }

    function sceneList() {
        return scenes.split(';');
    }

    function clear() {
        scenes = "";
    }
}
