#project(cuvistaGui)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

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
    player.cpp
    PlayerWidget.cpp
    progress.cpp
    UserInputGui.cpp
    ImageLabel.cpp
    ClickLabel.cpp
)

target_include_directories(cuvistaGui PRIVATE "${PROJECT_SOURCE_DIR}/Deshaker")
target_include_directories(cuvistaGui PRIVATE "${PROJECT_SOURCE_DIR}/cuvistaGui")

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
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
)
