
.NET bindings for Tensorflow 2.4 32 bits. Not all operations are supported, just inference. 64 bits is not supported. .NET dll target version is .NET 3.5

## Use
See the "TestApplication" project

```C#
int nThreads = 2;
ModelWrapper model = new ModelWrapper(@"path\to\model.tflite", nThreads);
foreach(TensorWrapper tensor in model.InputTensors)
{
	// Input is expected to be a one dimension array. If input has multiple
	// dimensions, all dimensions must to be combined in a single array
	int dim = tensor.Dimensions[0];
	tensor.SetValues(new Int32[dim]);
}
model.InvokeInterpreter();
foreach (TensorWrapper tensor in model.OutputTensors)
{
	// Output is expected to be a one dimension array. If real output has 
	// multiple dimensions, you must to reshape the returned array
	int dim = tensor.Dimensions[0];
	float[] output = new float[dim];
	tensor.GetValues(output);
	Console.WriteLine(tensor.Name + ": " + string.Join(", ", output));
}
```

## Building

My requirements are to use the bindings in a .NET 3.5 project, so, 
it's complicated:

* The VS solution in this repo is for VS 2019, and targeting .NET 3.5 is UNSUPPORTED (see 
  https://developercommunity.visualstudio.com/t/issue-with-ccli-targeting-net-framework-35/1182995
  ). So, you need VS 2008 (!!!) to be installed, and use it as C++ compiler
* VS 2008 does not include stdint.h (!!!) so, download it and copy it in 
  C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\include (see
  https://stackoverflow.com/questions/126279/c99-stdint-h-header-and-ms-visual-studio)

If you are targeting a .NET version >= 4.0 and you don't have VS 2008, you can use the
the VS 2019 C++ compiler changing the TfLiteNetWrapper project property "Platform toolset"

Common steps for any .NET target:

* Clone this repo.
* Clone [Tensorflow repo](https://github.com/tensorflow/tensorflow). Tested with code in master at commit [fed1cf9bc0d107eaa028bc6c375f750404ede5f0](https://github.com/tensorflow/tensorflow/tree/fed1cf9bc0d107eaa028bc6c375f750404ede5f0)
* Change TfLiteNetWrapper to point the cloned TF repo (TfLiteNetWrapper project properties > VC++ Directories > Include directories)
* Build solution
* To use these bindings you will need the generated TfLiteNetWrapper.dll and the TF Lite 32 bit binaries, included in tfliteLibraries/32bits/tensorflowlite_c.dll

## Test application
To run the test application:
* Change the path to model.tflite in Program.cs to your local path
* Copy tfliteLibraries/32bits/tensorflowlite_c.dll to the Debug/Release directory

## Licensing
This project contains a compiled version of Tensorflow Lite. See https://github.com/tensorflow/tensorflow for its license.
