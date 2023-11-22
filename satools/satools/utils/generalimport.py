"""
Handle optional dependencies with a single call.
From https://github.com/ManderaGeneral/generalimport

Example:
    generalimport("fakepackage")
    generalimport("scipy")
    import numpy
    import fakepackage # NO error
    from scipy import signal
    print(numpy.__path__)
    print(signal.__path__)
    print(fakepackage) # MissingOptionalDependency: Optional dependency 'fakepackage' was used but it isn't installed.
"""
import sys
from logging import getLogger
import importlib
import importlib.util
from pathlib import Path


def is_imported(module_name: str) -> bool:
    """
    Returns True if the module was actually imported, False, if generalimport mocked it.
    """
    module = sys.modules.get(module_name)
    try:
        return bool(module and not isinstance(module, FakeModule))
    except MissingOptionalDependency as exc:
        # isinstance() raises MissingOptionalDependency: fake module
        pass
    return False


class GeneralImporter:
    """ Creates fake packages if they don't exist.
        These fake packages' attrs are always a function that raises a ModuleNotFoundError when used.
        This lets you write a single line to handle all your optional dependencies.
        If wildcard (default "*") is provided then this will work on any missing package. """

    singleton_instance = None

    def __init__(self):
        self.catchers = []

        self._singleton()
        self._skip_fullname = None
        sys.meta_path.insert(0, self)

    def catch(self, fullname):
        """ Return first catcher that handles given fullname and filename.

            :rtype: generalimport.ImportCatcher """
        for catcher in self.catchers:
            if catcher.handle(fullname=fullname):
                return catcher

    def find_spec(self, fullname, path=None, target=None):
        if self._ignore_next_import(fullname=fullname):
            return self._handle_ignore(fullname=fullname, reason="Recursive break")

        if self._ignore_existing_top_name(fullname=fullname):
            return self._handle_ignore(fullname=fullname, reason="Top name exists and is not namespace")

        spec = get_spec(fullname)
        if not spec:
            return self._handle_handle(fullname=fullname, reason="Doesn't exist")

        if spec_is_namespace(spec=spec):
            return self._handle_handle(fullname=fullname, reason="Namespace package")

        return self._handle_relay(fullname=fullname, spec=spec)

    def create_module(self, spec):
        return FakeModule(spec=spec)

    def exec_module(self, module):
        pass

    def _singleton(self):
        assert self.singleton_instance is None
        GeneralImporter.singleton_instance = self

    def _ignore_existing_top_name(self, fullname):
        name = _get_top_name(fullname=fullname)
        if name == fullname:
            return False
        module = sys.modules.get(name, None)
        module_is_real = not fake_module_check(module, error=False)
        return module_is_real and not module_is_namespace(module)

    def _ignore_next_import(self, fullname):
        if fullname == self._skip_fullname:
            self._skip_fullname = None
            return True
        else:
            self._skip_fullname = fullname
            return False

    def _handle_ignore(self, fullname, reason):
        getLogger(__name__).debug(f"Ignoring '{fullname}' - {reason}")
        return None

    def _handle_handle(self, fullname, reason):
        catcher = self.catch(fullname=fullname)
        if not catcher:
            return self._handle_ignore(fullname=fullname, reason="Unhandled")

        getLogger(__name__).debug(f"{catcher} is handling '{fullname}' - {reason}")

        sys.modules.pop(fullname, None)  # Remove possible namespace

        return importlib.util.spec_from_loader(fullname, self)

    def _handle_relay(self, fullname, spec):
        getLogger(__name__).debug(f"'{fullname}' exists, returning it's spec '{spec}'")
        return spec


class FakeModule:
    """ Behaves like a module but any attrs asked for always returns self.
        Raises a ModuleNotFoundError when used in any way.
        Unhandled use-cases: https://github.com/ManderaGeneral/generalimport/issues?q=is%3Aissue+is%3Aopen+label%3Aunhandled """
    __path__ = []

    def __init__(self, spec):
        self.name = spec.name

        self.__name__ = spec.name
        self.__loader__ = spec.loader
        self.__spec__ = spec
        self.__fake_module__ = True  # Should not be needed, but let's keep it for safety?

    def error_func(self, *args, **kwargs):
        name = f"'{self.name}'" if hasattr(self, "name") else ""  # For __class_getitem__
        raise MissingOptionalDependency(f"Optional dependency {name} was used but it isn't installed.")

    def __getattr__(self, item):
        if item in self.non_called_dunders:
            self.error_func()
        return self

    # Binary
    __ilshift__ = __invert__ = __irshift__ = __ixor__ = __lshift__ = __rlshift__ = __rrshift__ = __rshift__ = error_func

    # Callable
    __call__ = error_func

    # Cast
    __bool__ = __bytes__ = __complex__ = __float__ = __int__ = __iter__ = __hash__ = error_func

    # Compare
    __eq__ = __ge__ = __gt__ = __instancecheck__ = __le__ = __lt__ = __ne__ = __subclasscheck__ = error_func

    # Context
    __enter__ = __exit__ = error_func

    # Delete
    __delattr__ = __delitem__ = __delslice__ = error_func

    # Info
    __sizeof__ = __subclasses__ = error_func

    # Iterable
    __len__ = __next__ = __reversed__ = __contains__ = __getitem__ = __setitem__ = error_func

    # Logic
    __and__ = __iand__ = __ior__ = __or__ = __rand__ = __ror__ = __rxor__ = __xor__ = error_func

    # Lookup
    __class_getitem__ = __dir__ = error_func

    # Math
    __abs__ = __add__ = __ceil__ = __divmod__ = __floor__ = __floordiv__ = __iadd__ = __ifloordiv__ = __imod__ = __imul__ = __ipow__ = __isub__ = __itruediv__ = __mod__ = __mul__ = __neg__ = __pos__ = __pow__ = __radd__ = __rdiv__ = __rdivmod__ = __rfloordiv__ = __rmod__ = __rmul__ = __round__ = __rpow__ = __rsub__ = __rtruediv__ = __sub__ = __truediv__ = __trunc__ = error_func

    # Matrix
    __imatmul__ = __matmul__ = __rmatmul__ = error_func

    # Object
    __init_subclass__ = __prepare__ = __set_name__ = error_func

    # Pickle
    __getnewargs__ = __getnewargs_ex__ = __getstate__ = __reduce__ = __reduce_ex__ = error_func

    # String
    __format__ = __fspath__ = __repr__ = __str__ = error_func

    # Thread
    __aenter__ = __aexit__ = __aiter__ = __anext__ = __await__ = error_func

    # Version
    __version__ = "99.0"


    non_called_dunders = (
        # Callable
        "__annotations__", "__closure__", "__code__", "__defaults__", "__globals__", "__kwdefaults__",

        # Info
        "__bases__", "__class__", "__dict__", "__doc__", "__module__", "__name__", "__qualname__", "__all__", "__slots__",
    )


def _get_skip_base_classes():
    from unittest.case import SkipTest
    yield SkipTest

    try:
        from _pytest.outcomes import Skipped
        yield Skipped
    except ImportError:
        pass

class MissingOptionalDependency(*_get_skip_base_classes()):
    def __init__(self, msg=None):
        self.msg = msg

    def __repr__(self):
        if self.msg:
            return f"MissingOptionalDependency: {self.msg}"
        else:
            return f"MissingOptionalDependency"

    def __str__(self):
        return self.msg or "MissingOptionalDependency"


def fake_module_check(obj, error=True):
    """ Simple assertion to trigger error_func earlier if module isn't installed. """
    if type(obj).__name__ == "FakeModule":
        if error:
            obj.error_func()
        else:
            return True
    else:
        return False

def module_is_namespace(module):
    """ Returns if given module is a namespace. """
    return hasattr(module, "__path__") and getattr(module, "__file__", None) is None

def module_name_is_namespace(name):
    """ Checks if module's name is a namespace without adding it to sys.modules. """
    was_in_modules = name in sys.modules
    module = _safe_import(name=name)
    is_namespace = module_is_namespace(module=module)

    if was_in_modules:
        sys.modules.pop(name, None)

    return is_namespace

def _safe_import(name):
    try:
        return importlib.import_module(name=name)
    except (ModuleNotFoundError, TypeError, ImportError) as e:
        return None

def spec_is_namespace(spec):
    return spec and spec.loader is None


def _assert_no_dots(names):
    for name in names:
        assert "." not in name, f"Dot found in '{name}', only provide package without dots."

def get_importer():
    """ Return existing or new GeneralImporter instance. """
    return GeneralImporter.singleton_instance or GeneralImporter()

def generalimport(*names):
    """ Adds names to a new ImportCatcher instance.
        Creates GeneralImporter instance if it doesn't exist. """
    # print(get_previous_frame_filename())
    _assert_no_dots(names=names)
    catcher = ImportCatcher(*names)
    get_importer().catchers.append(catcher)
    return catcher

def _get_previous_frame_filename(depth):
    frame = sys._getframe(depth)
    files = ("importlib", "generalimport_bottom.py")

    while frame:
        filename = frame.f_code.co_filename
        frame_is_origin = all(file not in filename for file in files)
        if frame_is_origin:
            return filename
        frame = frame.f_back

def _get_scope_from_filename(filename):
    last_part = Path(filename).parts[-1]
    return filename[0:filename.index(last_part)]

def _get_top_name(fullname):
    return fullname.split(".")[0]

def get_spec(fullname):
    return importlib.util.find_spec(fullname)

class ImportCatcher:
    WILDCARD = "*"

    def __init__(self, *names):
        self.names = set(names)
        self.added_names = set()
        self.added_fullnames = set()
        self.enabled = True
        self._scope = self._get_scope()

        getLogger(__name__).debug(f"Created Catcher with names {self.names} and scope {self._scope}")

        self.latest_scope_filename = None

    def handle(self, fullname):
        if not self._handle_name(fullname=fullname):
            return False
        if not self._handle_scope():
            return False

        self._store_handled_name(fullname=fullname)
        return True

    @staticmethod
    def _get_scope():
        filename = _get_previous_frame_filename(depth=4)
        return _get_scope_from_filename(filename=filename)

    def _store_handled_name(self, fullname):
        name = _get_top_name(fullname=fullname)
        self.added_names.add(name)
        self.added_fullnames.add(fullname)

    def _handle_name(self, fullname):
        name = _get_top_name(fullname=fullname)
        if self.WILDCARD in self.names:
            return True
        if name in self.names:
            return True
        return False

    def _handle_scope(self):
        if self._scope is None:
            return True
        filename = _get_previous_frame_filename(depth=6)
        self.latest_scope_filename = filename
        return filename.startswith(self._scope)
