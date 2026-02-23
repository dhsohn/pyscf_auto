from run_queue import record_status_event
from run_opt_metadata import write_run_metadata


class RunMetadataRecorder:
    def write(self, run_metadata_path, metadata):
        write_run_metadata(run_metadata_path, metadata)

    def record_status(
        self,
        event_log_path,
        run_id,
        run_dir,
        status,
        *,
        previous_status=None,
        details=None,
    ):
        record_status_event(
            event_log_path,
            run_id,
            run_dir,
            status,
            previous_status=previous_status,
            details=details,
        )

