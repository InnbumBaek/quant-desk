# state/

Machine-local runtime state: the hash-chained audit log, order state machine
files, daily position snapshots and the local DuckDB/parquet store. Nothing
here is committed — positions and P&L are not public data, and the audit chain
must be verifiable on the machine that wrote it.
