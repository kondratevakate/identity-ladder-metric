"""Import torchvision when Windows Application Control blocks its compiled extension (`torchvision/_C.pyd`,
WinError 4551, seen 14 Nov 2026). The model definitions LPIPS and pytorch-fid use (AlexNet, VGG16, Inception-v3) are
pure Python on torch ops; only the detection operators (nms, roi_align) live in the blocked extension. When the
import fails on the missing `torchvision::nms` registration, the meta-registration module is replaced by an empty
module and torchvision is imported again; the detection operators then stay unavailable (calling one raises).
Nothing is changed when the extension loads normally. Vendor code is not edited.
"""
import sys
import types

STATE = {"shim": False}


def ensure():
    """Import torchvision (with the shim only if needed); returns the module."""
    try:
        import torchvision
        return torchvision
    except RuntimeError as e:
        if "torchvision::nms" not in str(e):
            raise
    for k in [k for k in sys.modules if k == "torchvision" or k.startswith("torchvision.")]:
        del sys.modules[k]
    sys.modules["torchvision._meta_registrations"] = types.ModuleType("torchvision._meta_registrations")
    import torchvision
    STATE["shim"] = True
    return torchvision
