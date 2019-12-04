import logging
import sys

log_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s -damped- (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
log_handler.setFormatter(formatter)
