using System;

namespace Vietasr
{
    public sealed class VietasrException : Exception
    {
        public VietasrException(string message) : base(message)
        {
        }
    }
}
