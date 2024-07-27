#project(cuvistaGui)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
#qt_standard_project_setup()

set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

qt_add_executable(cuvistaGui
    cuvistaGui.ui
    eula.ui
    player.ui
    progress.ui

    cuvistaGui.qrc

    cuvistaGui.cpp
    Eula.cpp
    main.cpp
    PlayerWidget.cpp
    progress.cpp
    UserInputGui.cpp
)

target_link_libraries(cuvistaGui PRIVATE
    Common
    Matrix 
    CudaStuff 
    Deshaker 
    OpenCL 
    PkgConfig::ffmpeg
    cuda
    cpu_features
    nvcuvid
    nvidia-encode
    Qt6::Widgets
)
