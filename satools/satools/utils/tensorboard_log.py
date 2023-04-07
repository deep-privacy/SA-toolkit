import logging
import datetime
import os

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.summary.writer import event_file_writer


class LogHandlerSummaryWriter(logging.Handler):

    def __init__(
        self,
        summary_writer,
        title="Logging: ",
        *args, **kwds
    ):
        """
        Logging handler that logs to a TensorBoard instance.

        Args:
            summary_writer (SummaryWriter): The summarywriter to log to.
            title (string): Title/tag to write to.

        Example:
            >>> handler = utils.LogHandlerSummaryWriter(SummaryWriter())
            >>> handler.setFormatter(logging.Formatter(logging.root.handlers[0].formatter._fmt))
            >>> logging.getLogger().addHandler(handler)
        """
        super().__init__(*args, **kwds)
        self.summary_writer = summary_writer
        self.title = title + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.accomulated_entries = ""

    def emit(self, record):
        log_entry = self.format(record)

        # Markdown  new lines:
        self.accomulated_entries += "  \n"
        self.accomulated_entries += log_entry.replace("\n", "  \n")


        self.summary_writer.add_text(
            self.title,
            self.accomulated_entries,
            1,
            1,
        )

        # SummaryWriter is append only, here we remove/delete previous entry from SummaryWriter by recreating a new, clean one
        # we want to only have the last add_text event where tag starts with 'Logging'

        file_writer_path = self.summary_writer.file_writer.event_writer._file_name
        self.summary_writer.file_writer.event_writer.close() # close the old writer

        # load event and delete the old writer
        loader = event_file_loader.EventFileLoader(file_writer_path)
        events = list(loader.Load())
        os.remove(file_writer_path)

        # re-create a writer at same dir
        writer_event = event_file_writer.EventFileWriter(os.path.dirname(file_writer_path))

        last_log = None
        for event in events:
            #  print(event)
            if event.HasField("summary"):
                istext = False
                if event.summary.value[0].HasField("metadata"):
                    if event.summary.value[0].metadata:
                        istext = event.summary.value[0].metadata.plugin_data.plugin_name == "text"

                if event.summary.value[0].tag.startswith("Logging") and istext:
                    last_log = event
                else:
                    writer_event.add_event(event)
        if last_log != None:
            #  print(last_log)
            writer_event.add_event(last_log)
        writer_event.flush()
        self.summary_writer.file_writer.event_writer = writer_event

