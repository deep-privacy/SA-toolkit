import sys
import json

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
