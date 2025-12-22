# check_model_reg.py
import importlib
import inspect
from pprint import pprint

MODULE = "models"   # or "models.model_registry" — set to the module that defines REGISTRY
m = importlib.import_module(MODULE)

keys = sorted(m.REGISTRY.keys())
print("\n✅ Registered model keys:")
pprint(keys)

print("\nDetailed info:")
for k in keys:
    entry = m.REGISTRY[k]
    kind = type(entry).__name__            # function, class, type, etc.

    # safe docstring first line (or empty)
    raw_doc = getattr(entry, "__doc__", None)
    doc_first = ""
    if raw_doc:
        lines = raw_doc.strip().splitlines()
        if lines:
            doc_first = lines[0].strip()

    # try to get a signature
    sig = "<no signature>"
    try:
        # inspect.signature works for functions and callable classes (__call__)
        sig = str(inspect.signature(entry))
    except (ValueError, TypeError):
        # fallback: for classes show ctor signature if available
        try:
            sig = str(inspect.signature(entry.__init__))
        except Exception:
            sig = "<signature unavailable>"

    # try to find source file / module
    src = None
    try:
        src = inspect.getsourcefile(entry) or inspect.getfile(entry)
    except Exception:
        src = getattr(entry, "__module__", "<unknown module>")

    print(f"- {k}: kind={kind}  sig={sig}")
    if doc_first:
        print(f"    doc: {doc_first}")
    print(f"    defined in: {src}")
