
Windows .NET bindings for Tensorflow 2.9

It's a Visual Studio project to generate a DLL with bindings to use the Tensorflow Lite C API.

Basically it's a modified version of 
tensorflow\lite\experimental\examples\unity\TensorFlowLitePlugin\Assets\TensorFlowLite\SDK\Scripts\Interpreter.cs
file of the Tensorflow source code

Compiled TF Flite C API Windows dll are in [tfliteLibraries](tfliteLibraries).

The compiled TF Lite DLL distributed here does not have Flex support. The only way I have
found to to compile targeting x86 platforms is with CMake, and it seems it does not
support Flex 
(see https://github.com/tensorflow/tensorflow/issues/40157#issuecomment-833297376).

## Use

See TestApplication/Program.cs for an example

## Licensing
See https://github.com/tensorflow/tensorflow for Tensorflow license
