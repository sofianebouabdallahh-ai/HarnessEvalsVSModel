-- Queryable projection of the traces, plus `cells` (aggregates with CIs) and `suite`/`mode` columns.
create table if not exists runs (
  run_id      text primary key,
  started_at  timestamptz not null,
  model       text not null,
  mode        text not null,
  manifest    jsonb not null
);

create table if not exists episodes (
  episode_id      bigserial primary key,
  episode_key     text unique not null,
  run_id          text not null references runs,
  ts              timestamptz not null,
  benchmark       text, suite text, task_id text,
  harness         text, harness_version text, seed int, mode text,
  success         bool, attacked bool, attack_success bool, refused bool,
  injection_seen  bool,
  turns int, tool_calls int, tool_errors int, input_tokens int, output_tokens int,
  usd             numeric(10,5), wall_ms int,
  tainted         bool default false,
  error           text,
  trace_path      text
);
create index if not exists episodes_run_harness_bench on episodes (run_id, harness, benchmark);
create index if not exists episodes_ts on episodes (ts);

create table if not exists scores (
  episode_id  bigint references episodes on delete cascade,
  scorer      text, name text, value double precision, label text, explanation text,
  primary key (episode_id, scorer, name)
);

create table if not exists cells (
  run_id      text references runs,
  harness     text, benchmark text, suite text, mode text, metric text,
  value       double precision, ci_lo double precision, ci_hi double precision,
  n_episodes  int, n_tasks int,
  primary key (run_id, harness, benchmark, suite, mode, metric)
);
