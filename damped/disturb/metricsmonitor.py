import torch
from .managed_service import ManagedMemory
from tensorboardX import SummaryWriter
from threading import Lock

class MetricsMonitor:
    """Class that tracks training progress of a distrub-ed toolkit
    """

    def __init__(self, tensorboard_dir):
        self.tensorboard_writter = SummaryWriter(tensorboard_dir, flush_secs=10, max_queue=1)
        self._mutex_add = Lock()

    def add_scalar(self, tag, scalar_value, interval_log=300):
        """Add scalar data to summary at a interval_log rate.
        scalar are reduced with a mean

        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            interval_log (int): Optional interval_log at which rate to log
        
        """

        map = ManagedMemory().metricsmonitor_values

        with self._mutex_add:

            if tag not in map:
                map[tag] = 0
                map[tag + "iter"] = 0

            map[tag] += float(scalar_value)
            map[tag + "iter"] += 1

            if map[tag + "iter"] % interval_log == 0:
                self._push_scalar(tag)

    def push_scalar_tag(self, tag):
        """Trigger the addition of scalar linked tag to summary

        Args:
            tag (string): Data identifier
        """

        map = ManagedMemory().metricsmonitor_values

        with self._mutex_add:
            if tag not in map:
                return

            if map[tag] == 0:
                return

            self._push_scalar(tag)

    def _push_scalar(self, tag):
        map = ManagedMemory().metricsmonitor_values

        if tag + "total" not in map:
            map[tag + "total"] = 0

        self.tensorboard_writter.add_scalar(tag, map[tag] / map[tag + "iter"], map[tag + "total"])
        map[tag] = 0
        map[tag + "iter"] = 0
        map[tag + "total"] += 1
