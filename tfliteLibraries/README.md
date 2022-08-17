Here are Tensorflow Lite C API, TF version 2.9, compiled dlls for Windows.

* release_32b_ruy: Release compilation for 32b, wih RUY support
* release_32b_xnnpack: Release compilation for 32b, wih XNNPACK support
* release_64b_ruy: Release compilation for 64b, wih RUY support
* release_64b_xnnpack: Release compilation for 64b, wih XNNPACK support

Testing with a Transformer model, RUY seems to be MUCH more faster than XNNPACK.

See [BUILD.md](BUILD.md) about how to compile TF Lite for windows.
