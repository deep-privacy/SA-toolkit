import logging
import datetime
import os


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
        self.accomulated_entries = log_entry.replace("\n", "  \n") + "  \n" + self.accomulated_entries

        self.summary_writer.add_text(
            self.title,
            self.accomulated_entries,
            1,
            1,
        )
