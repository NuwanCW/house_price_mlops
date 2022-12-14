from typing import Dict, List, Optional
from evidently import model_monitoring
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.model_monitoring import DataDriftMonitor
from evidently.runner.loader import DataLoader, DataOptions
import hashlib
import pandas
import datetime
import dataclasses
import logging
from prometheus_client import Gauge

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def getDriftMonitoringService(config):
    loader = DataLoader()
    logger.info(f"config: {config}")
    options = MonitoringServiceOptions(**config["service"])

    reference_data = loader.load(
        options.reference_path,
        DataOptions(
            date_column=config["data_format"].get("date_column", None),
            separator=config["data_format"]["separator"],
            header=config["data_format"]["header"],
        ),
    )
    logger.info(f"reference dataset loaded: {len(reference_data)} rows")
    svc = MonitoringService(
        reference_data,
        options=options,
        column_mapping=ColumnMapping(**config["column_mapping"]),
    )
    return svc


@dataclasses.dataclass
class MonitoringServiceOptions:
    reference_path: str
    min_reference_size: int
    use_reference: bool
    moving_reference: bool
    window_size: int
    calculation_period_sec: int
    monitors: List[str]


monitor_mapping = {"data_drift": DataDriftMonitor}


class MonitoringService:
    metric: Dict[str, Gauge]
    last_run: Optional[datetime.datetime]

    def __init__(
        self,
        reference: pandas.DataFrame,
        options: MonitoringServiceOptions,
        column_mapping: ColumnMapping = None,
    ):
        self.monitoring = model_monitoring.ModelMonitoring(
            monitors=[monitor_mapping[k]() for k in options.monitors], options=[]
        )

        if options.use_reference:
            self.reference = reference.iloc[: -options.window_size, :].copy()
            self.current = pandas.DataFrame()
        else:
            self.reference = reference.copy()
            self.current = pandas.DataFrame().reindex_like(reference).dropna()
        self.column_mapping = column_mapping
        self.options = options
        self.metrics = {}
        self.next_run_time = None
        self.new_rows = 0
        self.hash = hashlib.sha256(
            pandas.util.hash_pandas_object(self.reference).values
        ).hexdigest()
        self.hash_metric = Gauge("evidently:reference_dataset_hash", "", labelnames=["hash"])

    def iterate(self, new_rows: pandas.DataFrame):
        rows_count = new_rows.shape[0]

        self.current = self.current.append(new_rows, ignore_index=True)

        self.new_rows += rows_count
        current_size = self.current.shape[0]
        if self.new_rows < self.options.window_size < current_size:
            self.current.drop(
                index=list(range(0, current_size - self.options.window_size)),
                inplace=True,
            )
            self.current.reset_index(drop=True, inplace=True)

        if current_size < self.options.window_size:
            logger.info(
                f"Not enough data for measurement: {current_size} of {self.options.window_size}."
                f" Waiting more data"
            )
            return
        if self.next_run_time is not None and self.next_run_time > datetime.datetime.now():
            logger.info(f"Next run at {self.next_run_time}")
            return
        self.next_run_time = datetime.datetime.now() + datetime.timedelta(
            seconds=self.options.calculation_period_sec
        )
        # print(f"came here", self.reference, self.current, self.column_mapping)
        self.monitoring.execute(self.reference, self.current, self.column_mapping)
        self.hash_metric.labels(hash=self.hash).set(1)

        # print(f"ans {self.monitoring.metrics()}")
        for metric, value, labels in self.monitoring.metrics():
            metric_key = f"evidently:{metric.name}"
            found = self.metrics.get(metric_key)
            if not found:
                found = Gauge(
                    metric_key,
                    "",
                    () if labels is None else list(sorted(labels.keys())),
                )
                self.metrics[metric_key] = found
            if labels is None:
                found.set(value)
            else:
                found.labels(**labels).set(value)