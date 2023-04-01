import sys

class StdoutFilterOut:
    """
    usage example:

    ```
    from contextlib import redirect_stdout
    with redirect_stdout(StdoutFilterOut(ignore="can not import s3prl.", to_keep="wavlm")):
        self.feature_extract = torch.hub.load('s3prl/s3prl', self.feat_type)
    ```
    """

    def __init__(self, *_, ignore=None, to_keep=None):
        self.last_ignore = False
        self.ignore = ignore
        self.to_keep = to_keep
        self.terminal = sys.stdout

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
