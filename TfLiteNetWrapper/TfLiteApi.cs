using System;
using System.Runtime.InteropServices;
using System.Linq;

using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using System.Text;

namespace TfLiteNetWrapper
{
    /// <summary>
    /// Simple C# bindings for the experimental TensorFlowLite C API.
    /// </summary>
    public class Interpreter : IDisposable
    {
        public delegate void LogCallbackDelegate(string logMessage);

        private LogCallbackDelegate LogCallback;

        public struct Options : IEquatable<Options>
        {
            /// <summary>
            /// The number of CPU threads to use for the interpreter.
            /// </summary>
            public int threads;

            public LogCallbackDelegate LogCallback;

            public bool Equals(Options other)
            {
                return threads == other.threads;
            }
        }

        public struct TensorInfo
        {
            public string name { get; internal set; }
            public DataType type { get; internal set; }
            public int[] dimensions { get; internal set; }
            public QuantizationParams quantizationParams { get; internal set; }

            public override string ToString()
            {
                return string.Format("name: {0}, type: {1}, dimensions: {2}, quantizationParams: {3}",
                    name,
                    type,
                    "[" + string.Join(",", dimensions.Select(d => d.ToString()).ToArray()) + "]",
                    "{" + quantizationParams + "}");
            }
        }

        private TfLiteModel model = IntPtr.Zero;
        private TfLiteInterpreter interpreter = IntPtr.Zero;
        private TfLiteInterpreterOptions options = IntPtr.Zero;

        /// <summary>
        /// Copy of model content in unmanaged memory. Only used if model was created from a byte[] buffer
        /// </summary>
        private IntPtr ModelBuffer = IntPtr.Zero;

        public Interpreter(byte[] modelData, Options options = default(Options))
        {
            // I this this is wrong. TfLiteModelCreate caller is responsible to keep the modelData buffer alive (AND PINNED!)
            // (see https://github.com/tensorflow/tensorflow/issues/39253#issuecomment-628418012)
            // Here, GCHandle will go out of scope, and, then, modelData can be moved by GC in heap, and then Bad Things will happen
            // It seems keep a GCHandle pinned indefinitely is bad for GC heap compactation, so create a modelData copy in
            // unmanaged memory
            // GCHandle modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
            // IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();
            // model = TfLiteModelCreate(modelDataPtr, modelData.Length);

            // Allocate and copy modelData buffer to unmanaged memory
            int bufferSize = Marshal.SizeOf(modelData[0]) * modelData.Length;
            ModelBuffer = Marshal.AllocHGlobal(bufferSize);
            Marshal.Copy(modelData, 0 , ModelBuffer, modelData.Length);

            model = TfLiteModelCreate(ModelBuffer, bufferSize);
            if (model == IntPtr.Zero)
                throw new Exception("Failed to create TensorFlowLite Model");

            Setup(options);
        }

        public Interpreter(string modelFilePath, Options options = default(Options))
        {
            model = TfLiteModelCreateFromFile(modelFilePath);
            if (model == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Model");

            Setup(options);
        }

        private void Setup(Options options)
        {
            if (!options.Equals(default(Options)))
            {
                this.options = TfLiteInterpreterOptionsCreate();
                TfLiteInterpreterOptionsSetNumThreads(this.options, options.threads);
                if (options.LogCallback != null)
                {
                    LogCallback = options.LogCallback;
                    TfLiteInterpreterOptionsSetErrorReporter(this.options, this.TfLogCallback, IntPtr.Zero);
                }
            }

            interpreter = TfLiteInterpreterCreate(model, this.options);
            if (interpreter == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Interpreter");
        }

        public void Dispose()
        {
            if (interpreter != IntPtr.Zero) TfLiteInterpreterDelete(interpreter);
            interpreter = IntPtr.Zero;
            if (model != IntPtr.Zero) TfLiteModelDelete(model);
            model = IntPtr.Zero;
            if (options != IntPtr.Zero) TfLiteInterpreterOptionsDelete(options);
            options = IntPtr.Zero;

            if(ModelBuffer != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(ModelBuffer);
                ModelBuffer = IntPtr.Zero;
            }
        }

        public void Invoke()
        {
            ThrowIfError(TfLiteInterpreterInvoke(interpreter));
        }

        public int GetInputTensorCount()
        {
            return TfLiteInterpreterGetInputTensorCount(interpreter);
        }

        public void SetInputTensorData(int inputTensorIndex, Array inputTensorData)
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(inputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
            ThrowIfError(TfLiteTensorCopyFromBuffer(
                tensor, tensorDataPtr, Buffer.ByteLength(inputTensorData)));
        }

        public void ResizeInputTensor(int inputTensorIndex, int[] inputTensorShape)
        {
            ThrowIfError(TfLiteInterpreterResizeInputTensor(
                interpreter, inputTensorIndex, inputTensorShape, inputTensorShape.Length));
        }

        public void AllocateTensors()
        {
            ThrowIfError(TfLiteInterpreterAllocateTensors(interpreter));
        }

        public int GetOutputTensorCount()
        {
            return TfLiteInterpreterGetOutputTensorCount(interpreter);
        }

        public void GetOutputTensorData(int outputTensorIndex, Array outputTensorData)
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(outputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
            ThrowIfError(TfLiteTensorCopyToBuffer(
                tensor, tensorDataPtr, Buffer.ByteLength(outputTensorData)));
        }

        public TensorInfo GetInputTensorInfo(int index)
        {
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, index);
            return GetTensorInfo(tensor);
        }

        public TensorInfo GetOutputTensorInfo(int index)
        {
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, index);
            return GetTensorInfo(tensor);
        }

        /// <summary>
        /// Returns a string describing version information of the TensorFlow Lite library.
        /// TensorFlow Lite uses semantic versioning.
        /// </summary>
        /// <returns>A string describing version information</returns>
        public static string GetVersion()
        {
            return Marshal.PtrToStringAnsi(TfLiteVersion());
        }

        private static string GetTensorName(TfLiteTensor tensor)
        {
            return Marshal.PtrToStringAnsi(TfLiteTensorName(tensor));
        }

        private static TensorInfo GetTensorInfo(TfLiteTensor tensor)
        {
            int[] dimensions = new int[TfLiteTensorNumDims(tensor)];
            for (int i = 0; i < dimensions.Length; i++)
            {
                dimensions[i] = TfLiteTensorDim(tensor, i);
            }
            return new TensorInfo()
            {
                name = GetTensorName(tensor),
                type = TfLiteTensorType(tensor),
                dimensions = dimensions,
                quantizationParams = TfLiteTensorQuantizationParams(tensor),
            };
        }

        private static void ThrowIfError(int resultCode)
        {
            if (resultCode != 0) throw new Exception("TensorFlowLite operation failed.");
        }

        private void TfLogCallback(IntPtr userData, string fmt, IntPtr args)
        {
            // https://stackoverflow.com/questions/6694612/c-sharp-p-invoke-varargs-delegate-callback
            StringBuilder sb = new StringBuilder(_vscprintf(fmt, args) + 1);
            vsprintf(sb, fmt, args);

            string formattedMessage = sb.ToString();
            LogCallback(formattedMessage);
        }

        #region Externs

        private const string TensorFlowLibrary = "tensorflowlite_c";

        public enum DataType
        {
            NoType = 0,
            Float32 = 1,
            Int32 = 2,
            UInt8 = 3,
            Int64 = 4,
            String = 5,
            Bool = 6,
            Int16 = 7,
            Complex64 = 8,
            Int8 = 9,
            Float16 = 10,
        }

        public struct QuantizationParams
        {
            public float scale;
            public int zeroPoint;

            public override string ToString()
            {
                return string.Format("scale: {0} zeroPoint: {1}", scale, zeroPoint);
            }
        }

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteVersion();

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelCreate(IntPtr model_data, int model_size);

        [DllImport(TensorFlowLibrary, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe TfLiteInterpreter TfLiteModelCreateFromFile([MarshalAs(UnmanagedType.LPStr)] string model_path);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelDelete(TfLiteModel model);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsCreate();

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions options);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetNumThreads(
            TfLiteInterpreterOptions options,
            int num_threads
        );

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteInterpreterCreate(
            TfLiteModel model,
            TfLiteInterpreterOptions optional_options);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterDelete(TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterGetInputTensorCount(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetInputTensor(
            TfLiteInterpreter interpreter,
            int input_index);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterResizeInputTensor(
            TfLiteInterpreter interpreter,
            int input_index,
            int[] input_dims,
            int input_dims_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterAllocateTensors(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterInvoke(TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterGetOutputTensorCount(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetOutputTensor(
            TfLiteInterpreter interpreter,
            int output_index);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe DataType TfLiteTensorType(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteTensorNumDims(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern int TfLiteTensorDim(TfLiteTensor tensor, int dim_index);

        [DllImport(TensorFlowLibrary)]
        private static extern uint TfLiteTensorByteSize(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteTensorName(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe QuantizationParams TfLiteTensorQuantizationParams(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteTensorCopyFromBuffer(
            TfLiteTensor tensor,
            IntPtr input_data,
            int input_data_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteTensorCopyToBuffer(
            TfLiteTensor tensor,
            IntPtr output_data,
            int output_data_size);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void ErrorReporter(IntPtr userData, [In][MarshalAs(UnmanagedType.LPStr)] string fmt, IntPtr args);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetErrorReporter(TfLiteInterpreterOptions options, ErrorReporter reporter,
            IntPtr userData);

        [DllImport("msvcrt.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int vsprintf(StringBuilder buffer, string format, IntPtr args);

        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int _vscprintf(string format, IntPtr ptr);

        #endregion
    }
}
