using System;
using TfLiteNetWrapper;

namespace TestApplication
{
	class Program
	{
		static void Main(string[] args)
		{
			try
			{
				ModelWrapper model = new ModelWrapper(@"D:\kbases\subversion\fuentesTensorflow2.4\pruebasDlls\ConsoleApplication1\Debug\model.tflite", 2);

				Int32[] values = new Int32[64];
				model.InputTensors[0].SetValues(values);

				Console.WriteLine("Done");
			}
			catch(Exception ex)
			{
				Console.WriteLine(ex.ToString());
			}
		}
	}
}
