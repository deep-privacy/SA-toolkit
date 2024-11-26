import sys
import json

def safe_gpu(n=1):
    from safe_gpu import safe_gpu
    safe_gpu.claim_gpus(n)


class SimpleNamespace:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        if isinstance(self, SimpleNamespace) and isinstance(other, SimpleNamespace):
           return self.__dict__ == other.__dict__
        return NotImplemented

    def __getattr__(self, key):
        if key not in self.__dict__:
            return None
        else:
            return self.__dict__[key]

    def __getitem__(self, key):
        if key not in self.__dict__:
            return None
        else:
            return self.__dict__[key]


class StdFilterOut:
    """
    usage example:

    ```
    from contextlib import redirect_stdout
    with redirect_stdout(StdFilterOut(ignore="can not import s3prl.", to_keep="wavlm")):
        self.feature_extract = torch.hub.load('s3prl/s3prl', self.feat_type)
    ```
    """

    def __init__(self, *_, ignore=None, to_keep=None, terminal=sys.stdout):
        self.last_ignore = False
        self.ignore = ignore
        self.to_keep = to_keep
        self.terminal = terminal

    def write(self, txt):
        if (self.ignore and self.ignore in txt) and (
            self.to_keep and self.to_keep not in txt
        ):
            self.last_ignore = True
            return
        if self.last_ignore and txt == "\n":
            self.last_ignore = False
        else:
            self.terminal.write(txt)

    def flush(self):
        pass

def fix_json(json_str):

    # remove new lines and extra spaces
    json_str = json_str.replace("\n", "")
    json_str = json_str.strip()
    # remove trailing commas
    json_str = json_str.replace(",}", "}")
    json_str = json_str.replace(",]", "]")

    # replace single quotes with double quotes
    json_str = json_str.replace("'", "\"")

    # replace equals with colons
    json_str = json_str.replace("=", ":")

    # load the fixed JSON object
    return json.loads(json_str)

