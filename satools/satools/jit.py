
"""
This module is used to on the fly, reload a pytorch class model
to make it work in jit for training and saving even if there is a unsupported (by the jit) call.

This is useful if you want to train most for your network with jit while sitll allowing some python functions
to be interpreted by python DURING training.
If your model does not use python3 for inference, it can be saved and later used with jit.
(but only in eval)
"""

class JITmode(object):
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
            class_._instance.mode = "for_training"
            #  class_._instance.mode = "for_save"
        #  print("New", class_._instance.mode)
        return class_._instance

    def forTraining(self):
        self.mode = "for_training" # uses @torch.jit.ignore
        # Make sure satools is reloaded
        self.reload()
        self.reload()
        self.reload()

    def forSave(self):
        self.mode = "for_save" # uses @torch.jit.unused
        # Make sure satools is reloaded
        self.reload()
        self.reload()
        self.reload()

    def select(self, f):
        import torch
        #  print("GET:", self.mode)
        if self.mode == "for_training":
            return torch.jit.ignore(f)
        #  print("GET:", "UNused")
        return torch.jit.unused(f)

    def reload(self):
        # to apply modif, we need to reload the model definitions
        from sys import modules
        import importlib
        for module in set(modules.values()):
            if module.__name__.startswith("satools") and module.__name__ != 'satools.jit':
                importlib.reload(module)
