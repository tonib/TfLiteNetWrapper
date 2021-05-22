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
				ModelWrapper model = new ModelWrapper(@"D:\kbases\subversion\fuentesTensorflow2.4\pruebasDlls\ConsoleApplication1\Debug\model.tflite");

				Console.WriteLine("Done");
			}
			catch(Exception ex)
			{
				Console.WriteLine(ex.ToString());
			}
		}
	}
}
