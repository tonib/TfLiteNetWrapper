
Windows .NET bindings for Tensorflow 2.4.1

It's a Visual Studio project to generate a DLL with bindings to use the Tensorflow Lite C API.

Basically it's a modified version of 
tensorflow\lite\experimental\examples\unity\TensorFlowLitePlugin\Assets\TensorFlowLite\SDK\Scripts\Interpreter.cs
file of the Tensorflow source code

Bindings should work in x86 and x64, but only x86 has been tested. There is a dll of the TF Flite C API compiled for x86 
in tfliteLibraries/32bits/tensorflowlite_c.dll in  this repo.

The compiled TF Lite DLL distributed here does not have Flex support. The only way I have
found to to compile targeting x86 platforms is with CMake, and it seems it does not
support Flex 
(see https://github.com/tensorflow/tensorflow/issues/40157#issuecomment-833297376).

## Use

See TestApplication/Program.cs for an example

## Licensing
See https://github.com/tensorflow/tensorflow for Tensorflow license
