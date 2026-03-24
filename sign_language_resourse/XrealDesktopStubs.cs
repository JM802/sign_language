using UnityEngine;

// Desktop-safe fallbacks so the project can compile without XREAL packages.
namespace Unity.XR.XREAL
{
    public static class XrealDesktopStubMarker
    {
    }
}

namespace Unity.XR.XREAL.Samples
{
    public class RGBCameraExample : MonoBehaviour
    {
        public Texture y_texture;
        public Texture u_texture;
        public Texture v_texture;
    }
}
