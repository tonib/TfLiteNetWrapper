
.NET bindings for Tensorflow 2.4 32 bits. Not all operations are supported, just inference. 64 bits is not supported

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
