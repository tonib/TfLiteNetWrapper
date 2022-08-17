
Compile Tensorflow Lite for Windows is really painful.
Here is a log of steps followed:

Should be this:
```bash
cmake ../tensorflow_src/tensorflow/lite
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

### Try 1

Error:
```
D:\Repositorios\Git\Otros\tensorflow2.9\tensorflow_src\tensorflow\lite\delegates\external\external_delegate.cc(158,11): error C7555: el uso de los inicializadores designados requiere al menos "/std:c++20" [D:
\Repositorios\Git\Otros\tensorflow2.9\tflite_build\tensorflow-lite.vcxproj]
```

Solution:
It seems there is a pending pull request fixing it. Change by hand external_delegate.cc.
See 
https://issuemode.com/issues/tensorflow/tensorflow/76268970
and 
https://github.com/tensorflow/tensorflow/pull/55746/files#diff-1e55a2daf822bac752c8d2a5f49e6f80eb3aae8c11de70d24dd9946a71497bd0

### Try 2

Ok. Now compiles, but compiles a static library. I want a dll:

```
cmake -DCMAKE_BUILD_TYPE=Release -S ..\tensorflow_src\tensorflow\lite\c -B .
```
See https://github.com/tensorflow/tensorflow/issues/47166

### Try 3

This fails with:

```
CMake Error at CMakeLists.txt:63 (add_library):
  Cannot find source file:

    common.c

  Tried extensions .c .C .c++ .cc .cpp .cxx .cu .mpp .m .M .mm .ixx .cppm .h
  .hh .h++ .hm .hpp .hxx .in .txx .f .F .for .f77 .f90 .f95 .f03 .hip .ispc


CMake Error at CMakeLists.txt:63 (add_library):
  No SOURCES given to target: tensorflowlite_c
```

Solution: Compilation is working fine after changing common.c to common.cc in the CMakeLists.txt.

See https://github.com/tensorflow/tensorflow/issues/56125


### Try 4

```
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

Error:

```
LINK : fatal error LNK1104: no se puede abrir el archivo 'm.lib' [D:\Repositorios\Git\Otros\tensorflow2.9\tflite_build\tensorflowlite_c.vcxproj]
```

Solution:

m.lib should be linked only in Linux, not in Windows.
Remove m.lib from linker list in VS project.

See https://stackoverflow.com/questions/54935559/linking-math-library-in-cmake-file-on-windows-and-linux


### Try 5

```
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

Compiles, but does not export any symbol. Solution: -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True ?

### Try 6

```
cmake ../tensorflow_src/tensorflow/lite/c -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True
cmake --build . -j
```

Compiles, exports symbols, but performance is very bad. 
Solution: Compile with Release

### Finally it works

```
# Compile 64b, with XNNPACK:
cmake ../tensorflow_src/tensorflow/lite/c -DCMAKE_BUILD_TYPE=Release -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True -DTFLITE_ENABLE_XNNPACK=ON
cmake --build . -j --config Release

# Compile 32b, with XNNPACK
# -j8 should make compilation faster, as it compiles concurrently. It did not worked for me
cmake ../tensorflow_src/tensorflow/lite/c -DCMAKE_BUILD_TYPE=Release -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True -DTFLITE_ENABLE_XNNPACK=ON -A Win32
cmake --build . -j --config Release -j8

# Compile 32b, with RUY
cmake ../tensorflow_src/tensorflow/lite/c -DCMAKE_BUILD_TYPE=Release -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True -DTFLITE_ENABLE_RUY=ON -A Win32
cmake --build . -j --config Release -j8

# Compile 64b, with RUY
cmake ../tensorflow_src/tensorflow/lite/c -DCMAKE_BUILD_TYPE=Release -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True -DTFLITE_ENABLE_RUY=ON
cmake --build . -j --config Release -j8
```






